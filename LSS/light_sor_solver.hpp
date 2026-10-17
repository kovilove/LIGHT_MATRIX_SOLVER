/**
 * @file light_sor_solver.hpp
 * @brief 轻量级 SOR（Successive Over-Relaxation，逐次超松弛）线性方程组求解器
 * @details 单头文件、header-only、C++17。
 *          CSR（Compressed Sparse Row）稀疏存储，适用于 CFD 等稀疏线性系统。
 *          用法：Reset → AddRow/AddMultiRow/AddMtx → SetRhs/AddRhs → SOR_Solve
 *          重填矩阵：ClearMtx → AddRow/AddMtx（保留尺寸与右端项）
 */
#ifndef LIGHT_SOR_SOLVER_H
#define LIGHT_SOR_SOLVER_H

#include <cmath>
#include <climits>
#include <cassert>

#include <memory>
#include <functional>
#include <type_traits>

#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace LSS
{
    // ====================== 求解器全局参数 ======================
    constexpr double DEFAULT_EPSILON = 1e-7;  // 判零阈值：|x| <= eps 视为 0
    constexpr double MAX_OVERLIMIT_TOL = 1e6; // 发散保护：绝对残差超过该值判定发散
    constexpr double MAX_SET_TOL = 1e-1;      // 收敛容差 tol 允许设置的最大值
    constexpr double DEFAULT_TOL = 1e-6;      // 默认收敛容差（基于归一化残差）
    constexpr double MAX_ALPHAW = 2.0;        // 松弛因子 alphaW 允许的最大值
    constexpr size_t MIN_ITER = 10;           // 迭代次数下限
    constexpr size_t MAX_ITER = 1000;         // 迭代次数上限

    /// @brief 求解器错误码
    enum ErrCode : unsigned short
    {
        LSS_OK = 0,              // 操作成功 / 收敛
        LSS_ERR,                 // 通用错误（保留）
        LSS_INI_ERR,             // 矩阵未初始化或初始化参数非法
        LSS_VEC_LEN_ERR,         // 向量长度不足
        LSS_MTX_SIZE_ERR,        // 矩阵大小不足
        LSS_DIAG_ELEM_ZERO,      // 对角元素为零（SOR 无法迭代）
        LSS_NAN_ELEM_ERR,        // 输入含 NaN/Inf
        LSS_SPARSE_ROW_FULL,     // 矩阵行已满，无法继续添加
        LSS_SPARSE_ROW_NOT_FULL, // 矩阵未装满，无法求解
        LSS_SOLVE_SET_ERR,       // 求解参数非法（iter/alphaW/tol 越界）
        LSS_NOT_DIAG_DOM,        // 不满足严格对角占优
        LSS_DIAG_DOM,            // 满足严格对角占优
        LSS_NOT_CONVERGED,       // 迭代跑满仍未收敛（x 不可信）
        LSS_DIVERGED,            // 迭代发散（残差爆炸 / NaN）
    };

    /**
     * @brief 判断元素是否接近零
     * @param cmpValue 待判断的元素值
     * @param epsilon 判零阈值
     * @return true 表示 |cmpValue| <= epsilon，视为零元素
     */
    template <typename T>
    inline static bool IsZeroElem(T cmpValue, T epsilon)
    {
        return std::abs(cmpValue) <= (epsilon);
    }

    /**
     * @brief 矩阵抽象基类：定义矩阵通用接口与公共状态
     * @note 禁止拷贝与移动（矩阵对象按引用使用）
     */
    class BaseMatrix
    {
    public:
        BaseMatrix() = default;
        virtual ~BaseMatrix() = default;
        BaseMatrix(const BaseMatrix &) = delete;
        BaseMatrix(BaseMatrix &&) = delete;
        BaseMatrix &operator=(const BaseMatrix &) = delete;
        BaseMatrix &operator=(BaseMatrix &&) = delete;

        /**
         * @brief 重置矩阵为 colSize×colSize 方阵并清空旧数据
         * @param colSize 矩阵列数（方阵，即行数），必须大于 2
         * @return LSS_OK 成功；LSS_INI_ERR 尺寸非法
         */
        virtual unsigned short Reset(size_t colSize = 0) = 0;

        /**
         * @brief 清空矩阵所有数据，恢复初始无效状态
         */
        virtual void Clear() = 0;

    protected:
        size_t colSize_ = {0};   // 矩阵列数
        size_t rowSize_ = {0};   // 当前已添加的行数
        bool isValid_ = {false}; // 是否已初始化（Reset 成功）
    };

    /**
     * @brief 基于 CSR 稀疏存储的方阵类，内置 SOR 迭代求解器
     * @tparam T 浮点类型（float/double），编译期 static_assert 校验
     * @note 非零元素按行压缩存储：value_/colIdx_/rowPtr_ 构成标准 CSR 三元组，
     *       diagIdx_ 记录每行对角元在 value_ 中的下标（求解时 O(1) 访问）
     */
    template <typename T>
    class SparseMatrix : public BaseMatrix
    {
        static_assert(std::is_floating_point_v<T>, "SparseMatrix requires floating-point T");

        using Arr = T *;
        using Vec = std::vector<T>;
        using Mtx = std::vector<std::vector<T>>;

        using CheckFunc = std::function<bool()>;
        struct CheckCondition
        {
            CheckFunc checkFunc = {nullptr};
            ErrCode errCode = {LSS_OK};
        };
        using CheckConditionLists = std::vector<CheckCondition>;

    public:
        SparseMatrix() = default;
        virtual ~SparseMatrix() = default;

        /**
         * @brief 重置矩阵为 colSize×colSize 方阵并清空旧数据
         * @param colSize 矩阵列数（方阵，即行数），必须大于 2
         * @return LSS_OK 成功；LSS_INI_ERR 尺寸非法（<= 2）
         * @note 按每行约 7 个非零元（如 3D CFD 7 点模板）预留内存
         */
        unsigned short Reset(size_t colSize = 0) override
        {
            // 清理旧资源
            Clear();

            // 太小的数组没有意义
            if (colSize <= 2)
            {
                return LSS_INI_ERR;
            }

            // 列数固定
            colSize_ = colSize;
            isValid_ = true;
            value_.reserve(colSize * 7);
            colIdx_.reserve(colSize * 7);
            rowPtr_.reserve(colSize + 1);
            diagIdx_.reserve(colSize);

            // 初始行数偏移为0
            rowPtr_.push_back(0);
            // 右端项大小固定
            rhs_.resize(colSize, T{});
            return LSS_OK;
        }

        /**
         * @brief 清空矩阵所有数据，恢复初始无效状态（判零阈值同步复位）
         */
        void Clear() override
        {
            colSize_ = 0;
            rowSize_ = 0;
            isValid_ = false;
            eps_ = static_cast<T>(DEFAULT_EPSILON);
            value_.clear();
            colIdx_.clear();
            rowPtr_.clear();
            diagIdx_.clear();
            rhs_.clear();
        }

        /**
         * @brief 清空矩阵系数内容（保留矩阵尺寸与右端项，可重新填充）
         * @note 仅清空 CSR 数据并重置行计数，colSize_/isValid_/rhs_ 保留；
         *       清空后可再次 AddRow/AddMtx 重新组装矩阵
         */
        void ClearMtx()
        {
            value_.clear();
            colIdx_.clear();
            rowPtr_.clear();
            diagIdx_.clear();
            rowSize_ = 0;

            value_.reserve(colSize_ * 7);
            colIdx_.reserve(colSize_ * 7);
            rowPtr_.reserve(colSize_ + 1);
            diagIdx_.reserve(colSize_);

            // 初始行数偏移为0
            rowPtr_.push_back(0);
        }

        /**
         * @brief 将右端项向量 b 全部清零（保留矩阵结构与尺寸）
         */
        void ClearRhs()
        {
            rhs_.assign(colSize_, T{});
        }

        /**
         * @brief 查询矩阵是否已初始化（Reset 成功且未 Clear）
         * @return true 已初始化
         */
        bool IsValid() const
        {
            return isValid_;
        }

        /**
         * @brief 查询矩阵是否已装满（已添加行数达到列数，可求解）
         * @return true 满矩阵，可调用 SOR_Solve
         */
        bool IsFullMatrix() const
        {
            return rowSize_ >= colSize_;
        }

        /**
         * @brief 获取矩阵列数
         * @return 列数（方阵，等于目标行数）
         */
        size_t ColSize() const
        {
            return colSize_;
        }

        /**
         * @brief 获取当前已添加的行数
         * @return 已添加行数（<= colSize_）
         */
        size_t RowSize() const
        {
            return rowSize_;
        }

        /**
         * @brief 获取矩阵非零元素总数
         * @return 非零元个数（CSR value_ 数组长度）
         */
        size_t NotZeroElemNum() const
        {
            return value_.size();
        }

        /**
         * @brief 获取右端项向量 b（只读）
         * @return 右端项 const 引用，避免拷贝
         */
        const Vec &Rhs() const
        {
            return rhs_;
        }

        /**
         * @brief 添加矩阵的一行（稠密输入，内部压缩为 CSR 存储）
         * @param vec 该行数据，长度必须 >= colSize，仅前 colSize 个元素有效
         * @return LSS_OK 成功；
         *         LSS_INI_ERR 矩阵未初始化；LSS_SPARSE_ROW_FULL 行已满；
         *         LSS_VEC_LEN_ERR 向量长度不足；
         *         LSS_DIAG_ELEM_ZERO 对角元为零；
         *         LSS_NAN_ELEM_ERR 元素含 NaN/Inf
         * @note 每行对角元（vec[rowSize_]）必须非零，否则 SOR 无法迭代；
         *       NaN/Inf 检查通过后才写入，失败不会留下部分数据
         */
        unsigned short AddRow(const Vec &vec)
        {
            auto result = AddRowAbnormalCheck(vec);
            if (result != LSS_OK)
            {
                return result;
            }

            for (size_t i = 0; i < colSize_; i++)
            {
                if (IsZeroElem(vec[i], eps_))
                {
                    continue;
                }
                value_.push_back(vec[i]);
                colIdx_.push_back(i);
                if (i == rowSize_)
                {
                    diagIdx_.push_back(colIdx_.size() - 1);
                }
            }
            rowPtr_.push_back(value_.size());
            rowSize_++;
            assert(rowPtr_.size() == (rowSize_ + 1));
            assert(diagIdx_.size() == rowSize_);
            return LSS_OK;
        }

        /**
         * @brief 追加批量行（等价于连续调用 AddRow，保留已有行）
         * @param mtx 待追加的稠密行数组（行向量数组）
         * @param errRow 输出参数：失败时返回出错行的行号（0 起始）
         * @return 同 AddRow
         * @note 与 AddMtx 的区别：AddMtx 会先清空矩阵再整体重建（覆盖语义），
         *       本函数在现有行之后继续追加；任一行失败时矩阵被 ClearMtx
         *       （保留尺寸与右端项，可修正后重试）
         */
        unsigned short AddMultiRow(const Mtx &mtx, size_t &errRow)
        {
            unsigned short result = LSS_OK;
            for (size_t i = 0; i < mtx.size(); i++)
            {
                result = AddRow(mtx[i]);
                if (result != LSS_OK)
                {
                    errRow = i;
                    ClearMtx();
                    return result;
                }
            }
            return LSS_OK;
        }

        /**
         * @brief 覆盖式批量组装矩阵（先清空现有行，再整体添加）
         * @param mtx 稠密矩阵（行向量数组），行数不能少于 colSize（超出的行按满行报错）
         * @param errRow 输出参数：失败时返回出错行的行号（0 起始）
         * @return LSS_MTX_SIZE_ERR 行数不足 colSize；其余同 AddRow
         * @note 开头先 ClearMtx 清空已有行（保留尺寸与右端项）；
         *       任一行失败时矩阵被 ClearMtx（保留尺寸与右端项，可修正后重试）
         */
        unsigned short AddMtx(const Mtx &mtx, size_t &errRow)
        {
            ClearMtx();
            // 检查是否能添加满
            if (mtx.size() < colSize_)
            {
                return LSS_MTX_SIZE_ERR;
            }
            unsigned short result = LSS_OK;
            for (size_t i = 0; i < mtx.size(); i++)
            {
                result = AddRow(mtx[i]);
                if (result != LSS_OK)
                {
                    errRow = i;
                    ClearMtx();
                    return result;
                }
            }
            return LSS_OK;
        }

        /**
         * @brief 累加右端项：rhs += vec（用于分步组装 b）
         * @param vec 待累加向量，长度必须 >= colSize
         * @return LSS_OK 成功；
         *         LSS_INI_ERR 未初始化；
         *         LSS_VEC_LEN_ERR 长度不足；
         *         LSS_NAN_ELEM_ERR 含 NaN/Inf
         */
        unsigned short AddRhs(const Vec &vec)
        {
            auto result = AddRowByVecAbnormalCheck(vec);
            if (result != LSS_OK)
            {
                return result;
            }

            for (size_t i = 0; i < colSize_; ++i)
            {
                rhs_[i] += vec[i];
            }
            return LSS_OK;
        }

        /**
         * @brief 右端项整体累加标量：rhs[i] += val（对所有 i）
         * @param val 待累加的标量
         * @return LSS_OK 成功；LSS_INI_ERR 未初始化；LSS_NAN_ELEM_ERR val 为 NaN/Inf
         */
        unsigned short AddRhs(T val)
        {
            auto result = AddRowByValueAbnormalCheck(val);
            if (result != LSS_OK)
            {
                return result;
            }

            for (size_t i = 0; i < colSize_; ++i)
            {
                rhs_[i] += val;
            }
            return LSS_OK;
        }

        /**
         * @brief 覆盖右端项：rhs = vec
         * @param vec 新右端项，长度必须 >= colSize
         * @return LSS_OK 成功；
         *         LSS_INI_ERR 未初始化；
         *         LSS_VEC_LEN_ERR 长度不足；
         *         LSS_NAN_ELEM_ERR 含 NaN/Inf
         */
        unsigned short SetRhs(const Vec &vec)
        {
            auto result = SetRhsByVecAbnormalCheck(vec);
            if (result != LSS_OK)
            {
                return result;
            }
            // copy vec -> rhs
            std::copy(vec.begin(), vec.begin() + colSize_, rhs_.begin());
            return LSS_OK;
        }

        /**
         * @brief 右端项整体覆盖为同一标量：rhs[i] = val（对所有 i）
         * @param val 待赋值的标量
         * @return LSS_OK 成功；LSS_INI_ERR 未初始化；LSS_NAN_ELEM_ERR val 为 NaN/Inf
         * @note 与 AddRhs(T) 的区别：本函数是覆盖语义，AddRhs(T) 是累加语义
         */
        unsigned short SetRhs(T val)
        {
            auto result = SetRhsByValueAbnormalCheck(val);
            if (result != LSS_OK)
            {
                return result;
            }
            // copy val -> rhs
            rhs_.assign(colSize_, val);
            return LSS_OK;
        }

        /**
         * @brief 使用逐次超松弛迭代（SOR）求解线性方程组 Ax = b
         * @param x 输入输出参数：进入时为零初值或上次解（作初值猜测），
         *          退出时为解向量（长度必须 >= colSize）
         * @param iter 最大迭代次数，合法范围 [MIN_ITER, MAX_ITER]
         * @param alphaW 松弛因子，合法范围 (0, 2]；
         *               SPD 矩阵取 (0, 2) 保证收敛，工程经验值 1.0 ~ 1.5，
         *               1.0 即 Gauss-Seidel
         * @param tol 收敛容差（归一化残差 ||Ax-b||/||b||），默认 1e-6，
         *            合法范围 (0, MAX_SET_TOL]
         * @return LSS_OK 收敛；
         *         LSS_INI_ERR 矩阵未初始化；
         *         LSS_VEC_LEN_ERR x长度不足；
         *         LSS_SPARSE_ROW_NOT_FULL 矩阵未填充满至方阵
         *         LSS_SOLVE_SET_ERR 入参iter/alphaW/tol设置错误（超限）
         *         LSS_NOT_CONVERGED 迭代跑满未达标（x 为最后迭代值，不可信）；
         *         LSS_DIVERGED 发散（NaN/Inf 或残差爆炸）；
         *         其余为参数/状态校验错误码
         * @note 每 10 次迭代检测一次残差；求解前自动预检严格对角占优（仅提示，不阻断）；
         *       收敛或发散时提前返回，x 保留当时的迭代结果
         */
        unsigned short SOR_Solve(Vec &x, size_t iter, T alphaW, T tol = static_cast<T>(DEFAULT_TOL))
        {
            auto result = SolverParaAbnormalCheck(x, iter, alphaW, tol);
            if (result != LSS_OK)
            {
                return result;
            }
            // 判断对角元素是否占优
            if (IsStrictlyDiagonallyDominant() != LSS_DIAG_DOM)
            {
                /* do something */
            }

            // 归一化残差
            T bNorm = ComputeUniformResidual();
            // 初始残差
            T res0 = ComputeResidual(x);

            const size_t checkEvery = 10;
            size_t startPos = 0;
            size_t endPos = 0;

            // 主迭代计算
            for (size_t it = 0; it < iter; it++)
            {
                for (size_t i = 0; i < colSize_; i++)
                {
                    startPos = rowPtr_[i];
                    endPos = rowPtr_[i + 1];
                    T rhs_i = rhs_[i];
                    T diagValue = value_[diagIdx_[i]];
                    for (size_t j = startPos; j < endPos; j++)
                    {
                        if (colIdx_[j] != i)
                        {
                            rhs_i -= value_[j] * x[colIdx_[j]];
                        }
                    }
                    x[i] = (1 - alphaW) * x[i] + alphaW * rhs_i / diagValue;
                }

                if ((it + 1) % checkEvery == 0)
                {
                    T res = ComputeResidual(x);
                    result = SolverResAbnormalCheck(res, res0, bNorm, tol);
                    if (result != LSS_OK)
                    {
                        return result;
                    }
                    // 计算是否提前达到收敛
                    T resNorm = IsZeroElem(bNorm, eps_) ? res : (res / bNorm);
                    if (resNorm < tol)
                    {
                        return LSS_OK; // 收敛，提前返回
                    }
                }
            }
            return LSS_NOT_CONVERGED;
        }

    private:
        /**
         * @brief 依次执行校验条件列表，任一条件成立即返回其对应错误码
         * @param checkLists 条件-错误码列表（lambda 返回 true 表示"校验失败"）
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short Check(const CheckConditionLists &checkLists) const
        {
            for (const auto &[checkFunc, errCode] : checkLists)
            {
                if ((checkFunc == nullptr) || checkFunc())
                {
                    return errCode;
                }
            }
            return LSS_OK;
        }

        /**
         * @brief 添加矩阵的一行前的异常检查
         * @param vec 待添加的行向量
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short AddRowAbnormalCheck(const Vec &vec)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_);
                 },
                 LSS_INI_ERR},
                {[&]() -> bool
                 {
                     return this->rowSize_ >= this->colSize_;
                 },
                 LSS_SPARSE_ROW_FULL},
                {[&]() -> bool
                 {
                     return vec.size() < this->colSize_;
                 },
                 LSS_VEC_LEN_ERR},
                {[&]() -> bool
                 {
                     return IsZeroElem(vec[this->rowSize_], this->eps_);
                 },
                 LSS_DIAG_ELEM_ZERO},
                {[&]() -> bool
                 {
                     for (size_t i = 0; i < this->colSize_; ++i)
                     {
                         if (std::isnan(vec[i]) || std::isinf(vec[i]))
                         {
                             return true;
                         }
                     }
                     return false;
                 },
                 LSS_NAN_ELEM_ERR},
            });
        }
        /**
         * @brief 添加矩阵的一行前的异常检查（仅检查向量长度和 NaN/Inf）
         * @param vec 待添加的行向量
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         * @note 仅用于累加右端项 rhs += vec 的异常检查，矩阵行数不受影响
         */
        unsigned short AddRowByVecAbnormalCheck(const Vec &vec)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_) ||
                            (this->rhs_.size() != this->colSize_);
                 },
                 LSS_INI_ERR},
                {[&]() -> bool
                 {
                     return vec.size() < this->colSize_;
                 },
                 LSS_VEC_LEN_ERR},
                {[&]() -> bool
                 {
                     for (size_t i = 0; i < this->colSize_; ++i)
                     {
                         if (std::isnan(vec[i]) || std::isinf(vec[i]))
                         {
                             return true;
                         }
                     }
                     return false;
                 },
                 LSS_NAN_ELEM_ERR},
            });
        }
        /**
         * @brief 添加矩阵的一个元素前的异常检查
         * @param val 待添加的元素值
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short AddRowByValueAbnormalCheck(T val)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_) ||
                            (this->rhs_.size() != this->colSize_);
                 },
                 LSS_INI_ERR},
                {[&]() -> bool
                 {
                     return std::isnan(val) || std::isinf(val);
                 },
                 LSS_NAN_ELEM_ERR},
            });
        }
        /**
         * @brief 设置右端项前的异常检查
         * @param vec 待设置的右端项向量
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short SetRhsByVecAbnormalCheck(const Vec &vec)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_) ||
                            (this->rhs_.size() != this->colSize_);
                 },
                 LSS_INI_ERR},
                {[&]() -> bool
                 {
                     return vec.size() < this->colSize_;
                 },
                 LSS_VEC_LEN_ERR},
                {[&]() -> bool
                 {
                     for (size_t i = 0; i < this->colSize_; ++i)
                     {
                         if (std::isnan(vec[i]) || std::isinf(vec[i]))
                         {
                             return true;
                         }
                     }
                     return false;
                 },
                 LSS_NAN_ELEM_ERR},
            });
        }
        /**
         * @brief 设置右端项前的异常检查（仅检查 NaN/Inf）
         * @param val 待设置的右端项标量
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short SetRhsByValueAbnormalCheck(T val)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_) ||
                            (this->rhs_.size() != this->colSize_);
                 },
                 LSS_INI_ERR},
                {[&]() -> bool
                 {
                     return std::isnan(val) || std::isinf(val);
                 },
                 LSS_NAN_ELEM_ERR},
            });
        }
        /**
         * @brief SOR 求解前的异常检查
         * @param x 解向量
         * @param iter 最大迭代次数
         * @param alphaW 松弛因子
         * @param tol 收敛容差
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short SolverParaAbnormalCheck(const Vec &x, size_t iter, T alphaW, T tol)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_) ||
                            (this->rhs_.size() != this->colSize_);
                 },
                 LSS_INI_ERR},
                {[&]() -> bool
                 {
                     return x.size() < this->colSize_;
                 },
                 LSS_VEC_LEN_ERR},
                {[&]() -> bool
                 {
                     return this->rowSize_ < this->colSize_;
                 },
                 LSS_SPARSE_ROW_NOT_FULL},
                {[&]() -> bool
                 {
                     return (iter < this->minIter_) || (iter > this->maxIter_) ||
                            (alphaW <= 0) || (alphaW > this->maxAlphaW_) ||
                            (tol <= 0) || (tol > static_cast<T>(MAX_SET_TOL));
                 },
                 LSS_SOLVE_SET_ERR},
            });
        }

        /**
         * @brief 检查矩阵是否严格对角占优（|Aii| > Σ|Aij|，j≠i）
         * @return LSS_DIAG_DOM 严格占优；
         *         LSS_NOT_DIAG_DOM 不占优；
         *         LSS_INI_ERR/LSS_SPARSE_ROW_NOT_FULL 状态非法
         * @note 严格对角占优是 SOR 收敛的充分条件而非必要条件，
         *       不占优不代表发散，仅提示风险
         */
        unsigned short IsStrictlyDiagonallyDominant() const
        {
            auto result = Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_) ||
                            (this->rhs_.size() != this->colSize_);
                 },
                 LSS_INI_ERR},
                {[&]() -> bool
                 {
                     return this->rowSize_ < this->colSize_;
                 },
                 LSS_SPARSE_ROW_NOT_FULL},
            });
            if (result != LSS_OK)
            {
                return result;
            }

            size_t startPos = 0;
            size_t endPos = 0;
            for (size_t i = 0; i < colSize_; i++)
            {
                startPos = rowPtr_[i];
                endPos = rowPtr_[i + 1];
                T diagValue = std::abs(value_[diagIdx_[i]]);
                T valueSum = {};
                for (size_t j = startPos; j < endPos; j++)
                {
                    if (colIdx_[j] != i)
                    {
                        valueSum += std::abs(value_[j]);
                    }
                }
                if (diagValue < valueSum - eps_)
                {
                    return LSS_NOT_DIAG_DOM;
                }
            }
            return LSS_DIAG_DOM;
        }

        /**
         * @brief SOR 求解迭代时的异常检查
         * @param res 当前残差
         * @param res0 初始残差
         * @param bNorm 均匀残差 L2 范数 ||b||₂
         * @param tol 收敛容差
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short SolverResAbnormalCheck(T res, T res0, T bNorm, T tol)
        {
            return Check({
                {[&]() -> bool
                 {
                     return std::isnan(res) || std::isinf(res);
                 },
                 LSS_DIVERGED},
                {[&]() -> bool
                 {
                     return res > static_cast<T>(MAX_OVERLIMIT_TOL);
                 },
                 LSS_DIVERGED},
                {[&]() -> bool
                 {
                     return (res0 > eps_) && (res > 1000 * res0);
                 },
                 LSS_DIVERGED},
            });
        }

        /**
         * @brief 计算均匀残差 L2 范数 ||b||₂
         * @return 均匀残差 L2 范数
         */
        T ComputeUniformResidual()
        {
            T res = T{};
            for (size_t i = 0; i < colSize_; i++)
            {
                res += rhs_[i] * rhs_[i];
            }
            return std::sqrt(res);
        }

        /**
         * @brief 计算当前解的绝对残差 L2 范数 ||Ax - b||₂
         * @param x 解向量
         * @return 残差 L2 范数
         * @note 调用前需保证矩阵已装满（rowSize == colSize），
         *       否则 rowPtr_ 越界（内部函数，由调用方保证）
         */
        T ComputeResidual(const Vec &x) const
        {
            T res = T{};
            size_t startPos = 0;
            size_t endPos = 0;
            for (size_t i = 0; i < colSize_; i++)
            {
                startPos = rowPtr_[i];
                endPos = rowPtr_[i + 1];
                T rhs_i = rhs_[i];
                for (size_t j = startPos; j < endPos; j++)
                {
                    rhs_i -= value_[j] * x[colIdx_[j]];
                }
                res += rhs_i * rhs_i;
            }
            return std::sqrt(res);
        }

    private:
        std::vector<T> value_;                             // 非零元素值（CSR）
        std::vector<size_t> colIdx_;                       // 非零元素列索引（CSR）
        std::vector<size_t> rowPtr_;                       // 每行起始偏移（CSR，长度 = 行数+1）
        std::vector<size_t> diagIdx_;                      // 每行对角元在 value_ 中的下标
        std::vector<T> rhs_;                               // 右端项 b
        T eps_ = static_cast<T>(DEFAULT_EPSILON);          // 判零阈值
        const size_t minIter_ = {MIN_ITER};                // 最小迭代次数
        const size_t maxIter_ = {MAX_ITER};                // 最大迭代次数
        const T maxAlphaW_ = {static_cast<T>(MAX_ALPHAW)}; // 松弛因子上限
    };
}

#endif // LIGHT_SOR_SOLVER_H
