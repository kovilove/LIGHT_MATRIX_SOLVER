/**
 * @file light_sor_solver.hpp
 * @brief 轻量级 SOR（Successive Over-Relaxation，逐次超松弛）线性方程组求解器
 * @details 单头文件、header-only、C++17。
 *          CSR（Compressed Sparse Row）稀疏存储，适用于 CFD 等稀疏线性系统。
 *          数据（SparseMatrix）与求解算法（LinearSolverBase / SORSolver）解耦：
 *          矩阵只负责存储与组装，求解器负责迭代算法，支持可插拔扩展（未来可新增 GAMG 等）。
 *          用法：Reset → AddRow/AddMultiRow/AddMtx → SetRhs/AddRhs → SORSolver(...).Solve(x, mat)
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
     * @brief 计算模式枚举
     * @note  串行模式（默认）/ 并行模式 / 自动模式（保留）
     */
    enum CalcProcessMode : unsigned short
    {
        SOR_SERIAL_MODE = 0,    // 串行模式
        SOR_MULTI_PROCESS_MODE, // 并行模式（红黑染色 SOR）
        SOR_AUTO_MODE,          // 自动模式（规模足够大且多线程时并行）
        SOR_MODE_MAX,
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
     * @brief 依次执行校验条件列表，任一条件成立即返回其对应错误码
     * @param checkLists 条件-错误码列表（lambda 返回 true 表示"校验失败"）
     * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
     */
    using CheckFunc = std::function<bool()>;
    struct CheckCondition
    {
        CheckFunc checkFunc = {nullptr};
        ErrCode errCode = {LSS_OK};
    };
    using CheckConditionLists = std::vector<CheckCondition>;
    unsigned short Check(const CheckConditionLists &checkLists)
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
     * @brief 矩阵抽象基类：定义矩阵通用接口与公共状态
     * @note 禁止拷贝与移动（矩阵对象按引用使用）
     */
    class BaseMatrix
    {
    public:
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
        BaseMatrix() = default;
        size_t colSize_ = {0};   // 矩阵列数
        size_t rowSize_ = {0};   // 当前已添加的行数
        bool isValid_ = {false}; // 是否已初始化（Reset 成功）
    };

    /**
     * @brief 基于 CSR 稀疏存储的方阵类（数据容器，仅负责存储与组装，不含迭代算法）
     * @tparam T 浮点类型（float/double），编译期 static_assert 校验
     * @note 非零元素按行压缩存储：value_/colIdx_/rowPtr_ 构成标准 CSR 三元组，
     *       diagIdx_ 记录每行对角元在 value_ 中的下标（求解时 O(1) 访问）
     * @note 求解算法见 LinearSolverBase / SORSolver；矩阵通过只读访问器
     *       （Values/ColIdx/RowPtr/DiagIdx/Rhs）向求解器暴露内部 CSR 结构。
     */
    template <typename T>
    class SparseMatrix : public BaseMatrix
    {
        static_assert(std::is_floating_point_v<T>, "SparseMatrix requires floating-point T");
        using Arr = T *;
        using Vec = std::vector<T>;
        using Mtx = std::vector<std::vector<T>>;

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
            structureRevision_ = 0;
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
            structureRevision_++; // 结构变化，求解器需重染色
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
         * @return true 满矩阵，可调用求解器 Solve
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

        // ====================== CSR 结构只读访问器（供求解器读取） ======================

        /**
         * @brief 获取 CSR 非零元值数组（只读）
         * @return value_ const 引用
         */
        const std::vector<T> &Values() const
        {
            return value_;
        }

        /**
         * @brief 获取 CSR 列索引数组（只读）
         * @return colIdx_ const 引用
         */
        const std::vector<size_t> &ColIdx() const
        {
            return colIdx_;
        }

        /**
         * @brief 获取 CSR 行偏移数组（只读，长度 = 行数 + 1）
         * @return rowPtr_ const 引用
         */
        const std::vector<size_t> &RowPtr() const
        {
            return rowPtr_;
        }

        /**
         * @brief 获取每行对角元在 value_ 中的下标（只读）
         * @return diagIdx_ const 引用
         */
        const std::vector<size_t> &DiagIdx() const
        {
            return diagIdx_;
        }

        /**
         * @brief 获取判零阈值
         * @return eps_
         */
        T Eps() const
        {
            return eps_;
        }

        /**
         * @brief 获取矩阵结构版本号
         * @details 每次结构变化（ClearMtx / AddRow）递增，用于求解器判断
         *          是否需重建红黑染色缓存等与结构相关的预处理数据
         * @return 当前结构版本号
         */
        size_t StructureRevision() const
        {
            return structureRevision_;
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
            structureRevision_++;
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
         * @brief 计算均匀残差 L2 范数 ||b||₂
         * @return 均匀残差 L2 范数
         */
        T ComputeUniformResidual() const
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

    private:
        std::vector<T> value_;                    // 非零元素值（CSR）
        std::vector<size_t> colIdx_;              // 非零元素列索引（CSR）
        std::vector<size_t> rowPtr_;              // 每行起始偏移（CSR，长度 = 行数+1）
        std::vector<size_t> diagIdx_;             // 每行对角元在 value_ 中的下标
        std::vector<T> rhs_;                      // 右端项 b
        T eps_ = static_cast<T>(DEFAULT_EPSILON); // 判零阈值
        size_t structureRevision_ = {0};          // 矩阵结构版本号（结构变化时递增）
    };

    /**
     * @brief 线性求解器抽象基类（策略模式）
     * @tparam T 浮点类型（float/double）
     * @details 定义统一求解接口，将"求解算法"与"矩阵数据"解耦。
     *          具体求解器（SOR / 未来的 GAMG / PCG 等）继承本类并实现 Solve。
     *          上层框架（如 CFD SIMPLE）只依赖本基类，可运行时切换求解器。
     */
    template <typename T>
    class LinearSolverBase
    {
        using Arr = T *;
        using Vec = std::vector<T>;
        using Mtx = std::vector<std::vector<T>>;

    public:
        virtual ~LinearSolverBase() = default;
        LinearSolverBase(const LinearSolverBase &) = delete;
        LinearSolverBase(LinearSolverBase &&) = delete;
        LinearSolverBase &operator=(const LinearSolverBase &) = delete;
        LinearSolverBase &operator=(LinearSolverBase &&) = delete;

        /**
         * @brief 求解线性方程组 A x = b（A 与 b 由矩阵 mat 提供）
         * @param x 输入输出参数：进入时为零初值或上次解（作初值猜测），
         *          退出时为解向量（长度必须 >= mat.ColSize()）
         * @param mat 系数矩阵与右端项（数据容器，求解器只读）
         * @return LSS_OK 收敛；其余为错误码（见 ErrCode）
         */
        virtual unsigned short Solve(Vec &x, const SparseMatrix<T> &mat) = 0;

        /// @brief 获取最后一次求解的最终残差（未求解时为 0）
        virtual T GetResidual() const = 0;

        /// @brief 获取最后一次求解实际执行的迭代次数
        virtual size_t GetIterCount() const = 0;

        /// @brief 查询最后一次求解是否收敛
        virtual bool IsConverged() const = 0;

    protected:
        LinearSolverBase() = default;
    };

    /**
     * @brief SOR（逐次超松弛）求解器实现
     * @tparam T 浮点类型（float/double）
     * @details 当前版本提供串行 SOR 迭代。红黑染色并行（OpenMP）将在后续
     *          版本加入：通过将未知量按矩阵拓扑二分为红/黑两组消除 GS/SOR
     *          迭代依赖，一轮迭代拆成两个并行 pass，保留顺序收敛语义。
     *          构造参数 mode 已为并行扩展预留（当前统一按串行执行）。
     */
    template <typename T>
    class SORSolver : public LinearSolverBase<T>
    {
        using Arr = T *;
        using Vec = std::vector<T>;
        using Mtx = std::vector<std::vector<T>>;

    public:
        SORSolver(size_t maxIter, T omega, T tol = static_cast<T>(DEFAULT_TOL),
                  CalcProcessMode mode = SOR_AUTO_MODE)
            : maxIter_(maxIter), omega_(omega), tol_(tol), mode_(mode),
              finalRes_(T{}), iterCnt_(0), converged_(false)
        {
            // 红黑染色并行将在后续版本启用，当前统一串行
        }

        /**
         * @brief 求解线性方程组 A x = b
         * @param x 输入输出参数：进入时为零初值或上次解（作初值猜测），
         *          退出时为解向量（长度必须 >= mat.ColSize()）
         * @param mat 系数矩阵与右端项（数据容器，求解器只读）
         * @return LSS_OK 收敛；
         *         LSS_INI_ERR 矩阵未初始化；
         *         LSS_VEC_LEN_ERR x长度不足；
         *         LSS_SPARSE_ROW_NOT_FULL 矩阵未填充满至方阵；
         *         LSS_SOLVE_SET_ERR 构造参数 iter/omega/tol 非法；
         *         LSS_NOT_CONVERGED 迭代跑满未达标（x 为最后迭代值，不可信）；
         *         LSS_DIVERGED 发散（NaN/Inf 或残差爆炸）；
         *         其余为参数/状态校验错误码
         * @note 每 10 次迭代检测一次残差；收敛或发散时提前返回，x 保留当时结果
         */
        unsigned short Solve(Vec &x, const SparseMatrix<T> &mat) override
        {
            auto result = SolverParaCheck(x, mat);
            if (result != LSS_OK)
            {
                return result;
            }

            // 归一化残差 / 初始残差
            T bNorm = mat.ComputeUniformResidual();
            T res0 = mat.ComputeResidual(x);

            // 当前版本统一串行；红黑染色并行（mode_）将在后续版本启用
            result = SolveSerial(x, mat, bNorm, res0);
            return result;
        }

        T GetResidual() const override
        {
            return finalRes_;
        }

        size_t GetIterCount() const override
        {
            return iterCnt_;
        }

        bool IsConverged() const override
        {
            return converged_;
        }

    private:
        /**
         * @brief 求解前参数与状态校验
         * @param x 解向量
         * @param mat 系数矩阵与右端项
         * @return LSS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short SolverParaCheck(const Vec &x, const SparseMatrix<T> &mat) const
        {
            return Check({
                {[&]() -> bool
                 {
                     return (!mat.IsValid()) ||
                            (mat.ColSize() == 0) ||
                            (mat.Rhs().size() != mat.ColSize());
                 },
                 LSS_INI_ERR},
                {[&]() -> bool
                 {
                     return x.size() < mat.ColSize();
                 },
                 LSS_VEC_LEN_ERR},
                {[&]() -> bool
                 {
                     return mat.RowSize() < mat.ColSize();
                 },
                 LSS_SPARSE_ROW_NOT_FULL},
                {[&]() -> bool
                 {
                     return ((maxIter_ < MIN_ITER) ||
                             (maxIter_ > MAX_ITER) ||
                             (omega_ <= 0 || omega_ > static_cast<T>(MAX_ALPHAW)) ||
                             (tol_ <= 0 || tol_ > static_cast<T>(MAX_SET_TOL)));
                 },
                 LSS_SOLVE_SET_ERR},
            });
        }

        /**
         * @brief 迭代过程中的发散检查
         * @param res 当前残差
         * @param res0 初始残差
         * @param bNorm 均匀残差 L2 范数 ||b||₂
         * @param eps 判零阈值
         * @return LSS_OK 正常；LSS_DIVERGED 发散
         */
        unsigned short DivergenceCheck(T res, T res0, T bNorm, T eps) const
        {
            return Check({
                {[&]() -> bool
                 {
                     return std::isnan(res) || std::isinf(res) ||
                            (res > static_cast<T>(MAX_OVERLIMIT_TOL)) ||
                            ((res0 > eps) && (res > 1000 * res0));
                 },
                 LSS_DIVERGED},
            });
        }

        /**
         * @brief 串行 SOR 迭代
         * @param x 解向量（原地更新）
         * @param mat 系数矩阵与右端项
         * @param bNorm 均匀残差 ||b||₂
         * @param res0 初始残差
         * @return 收敛/发散/未收敛错误码
         */
        unsigned short SolveSerial(Vec &x, const SparseMatrix<T> &mat, T bNorm, T res0)
        {
            const auto &value = mat.Values();
            const auto &colIdx = mat.ColIdx();
            const auto &rowPtr = mat.RowPtr();
            const auto &diagIdx = mat.DiagIdx();
            const auto &rhs = mat.Rhs();
            const size_t n = mat.ColSize();
            const T eps = mat.Eps();

            iterCnt_ = 0;
            converged_ = false;
            const size_t checkEvery = 10;

            for (size_t it = 0; it < maxIter_; it++)
            {
                for (size_t i = 0; i < n; i++)
                {
                    size_t start = rowPtr[i];
                    size_t end = rowPtr[i + 1];
                    T sum = rhs[i];
                    for (size_t j = start; j < end; j++)
                    {
                        if (colIdx[j] != i)
                        {
                            sum -= value[j] * x[colIdx[j]];
                        }
                    }
                    x[i] = (1 - omega_) * x[i] + omega_ * sum / value[diagIdx[i]];
                }

                if ((it + 1) % checkEvery == 0)
                {
                    T res = mat.ComputeResidual(x);
                    finalRes_ = res;
                    iterCnt_ = it + 1;
                    auto err = DivergenceCheck(res, res0, bNorm, eps);
                    if (err != LSS_OK)
                    {
                        return err;
                    }
                    // 归一化残差收敛判断
                    T resNorm = IsZeroElem(bNorm, eps) ? res : (res / bNorm);
                    if (resNorm < tol_)
                    {
                        converged_ = true;
                        return LSS_OK;
                    }
                }
            }
            return LSS_NOT_CONVERGED;
        }

    private:
        size_t maxIter_;       // 最大迭代次数
        T omega_;              // 松弛因子
        T tol_;                // 收敛容差
        CalcProcessMode mode_; // 计算模式（预留：红黑并行将在后续版本启用）

        T finalRes_;     // 最终残差
        size_t iterCnt_; // 实际迭代次数
        bool converged_; // 是否收敛
    };
}

#endif // LIGHT_SOR_SOLVER_H
