/**
 * @file lms_sparse_matrix.hpp
 * @brief 轻量级稀疏矩阵线性求解头文件库（header-only，C++17）
 * @details 基于 CSR（Compressed Sparse Row，压缩稀疏行）稀疏存储的线性方程组求解库，
 *          适用于 CFD 等大规模稀疏线性系统。
 *
 *          采用「数据 / 算法解耦 + 策略模式」的可插拔架构：
 *          - SparseMatrix：矩阵数据容器，只负责存储与组装，不含算法。
 *            通过 AddRow(vec, row) / AddValue 以 COO 坐标三元组动态装配，
 *            Finalize 统一排序、压缩并转换为标准 CSR 结构；
 *          - LinearSolverBase（见lms_linear_solver.hpp）：求解算法抽象基类，定义统一求解接口
 *            （Solve / 求解模式 / 收敛状态），供上层框架按基类指针运行时可切换求解器；
 *          - 具体求解器继承基类并实现 Solve：当前实现 SOR（逐次超松弛，含红黑染色并行）
 *            与 Jacobi（雅可比，含直接并行），未来可扩展 Krylov 子空间法
 *            （CG / BiCGSTAB / GMRES）、多重网格（AMG / GAMG）等，上层调用无需改动。
 *
 *          用法：
 *          - 装配矩阵：Reset → AddRow(vec, row) / AddValue → Finalize；
 *          - 重填矩阵：ClearMat → 重新装配（保留矩阵尺寸）；
 *          - 求解：solver.Solve(mat, x, rhs)（右端项由调用方持有）。
 */

#ifndef LMS_SPARSE_MATRIX_HPP
#define LMS_SPARSE_MATRIX_HPP

#include <cmath>
#include <climits>
#include <cassert>
#include <algorithm>

#include <memory>
#include <functional>
#include <type_traits>
#include <vector>

namespace LMS
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
        LMS_OK = 0,              // 操作成功 / 收敛
        LMS_ERR,                 // 通用错误（保留）
        LMS_INI_ERR,             // 矩阵未初始化或初始化参数非法
        LMS_VEC_LEN_ERR,         // 向量长度不足
        LMS_OUT_SIZE_LIMIT,      // 添加行/列数超限
        LMS_DIAG_ELEM_ZERO,      // 对角元素为零（SOR 无法迭代）
        LMS_NAN_ELEM_ERR,        // 输入含 NaN/Inf
        LMS_SPARSE_ROW_NOT_FULL, // 矩阵未装满，无法求解
        LMS_STATUS_REJECT,       // 当前装配状态不允许该操作
        LMS_COOTRIPLES_EMPTY,    // COO 三元组数组为空
        // 求解器相关错误码↓
        LMS_MARTIX_ERR,    // 求解矩阵不合法
        LMS_SOLVE_SET_ERR, // 求解参数非法（iter/alphaW/tol 越界）
        LMS_NOT_DIAG_DOM,  // 不满足严格对角占优
        LMS_DIAG_DOM,      // 满足严格对角占优
        LMS_NOT_CONVERGED, // 迭代跑满仍未收敛（x 不可信）
        LMS_DIVERGED,      // 迭代发散（残差爆炸 / NaN）
        LMS_MODE_ERR,      // 求解模式非法
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
     * @return LMS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
     */
    using CheckFunc = std::function<bool()>;
    /// @brief 单个校验项：校验函数 + 失败时对应的错误码
    struct CheckCondition
    {
        CheckFunc checkFunc = {nullptr}; // 校验函数（nullptr 视为校验失败）
        ErrCode errCode = {LMS_OK};      // 该校验失败时返回的错误码
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
        return LMS_OK;
    }

    // ====================== 存储数据矩阵 ======================
    /**
     * @brief 矩阵抽象基类：定义矩阵通用接口与公共状态
     * @note 禁止拷贝与移动（矩阵对象按引用使用）
     */
    class MatrixBase
    {
    public:
        virtual ~MatrixBase() = default;
        MatrixBase(const MatrixBase &) = delete;
        MatrixBase(MatrixBase &&) = delete;
        MatrixBase &operator=(const MatrixBase &) = delete;
        MatrixBase &operator=(MatrixBase &&) = delete;

        /**
         * @brief 重置矩阵为 colSize×colSize 方阵并清空旧数据
         * @param colSize 矩阵列数（方阵，即行数），必须大于 2
         * @return LMS_OK 成功；LMS_INI_ERR 尺寸非法
         */
        virtual unsigned short Reset(size_t colSize = 0) = 0;

        /**
         * @brief 清空矩阵所有数据，恢复初始无效状态
         */
        virtual void Clear() = 0;

    protected:
        MatrixBase() = default;
    };

    // ====================== CSR 稀疏存储矩阵 ======================
    /**
     * @brief 基于 CSR 稀疏存储的方阵类（数据容器，仅负责存储与组装，不含迭代算法）
     * @tparam T 浮点类型（float/double），编译期 static_assert 校验
     * @note 非零元素按行压缩存储：value_/colIdx_/rowPtr_ 构成标准 CSR 三元组，
     *       diagIdx_ 记录每行对角元在 value_ 中的下标（求解时 O(1) 访问）
     * @note 装配采用 COO 动态方式：以 AddRow(vec,row) / AddValue 按任意行序添加
     *       坐标三元组（同一 (row,col) 可重复累加），Finalize 统一排序、压缩并
     *       转换为标准 CSR 结构供求解器读取。
     * @note 求解算法见 LinearSolverBase / SORSolver；矩阵通过只读访问器
     *       （Values/ColIdx/RowPtr/DiagIdx/Eps/StructureRevision）向求解器暴露内部 CSR 结构；
     *       右端项 b 不存入矩阵，由调用方以 Solve(mat, x, rhs) 参数传入。
     */

    /**
     * @brief 矩阵装配状态机
     * @note INVALID：未初始化；INIT：已 Reset 待装配；WORKING：装配进行中；
     *       FINISHED：装配完成，可求解
     */
    enum SparseMatrixAssemblyStatus : unsigned short
    {
        LMS_ASSEMBLY_STATUS_INVALID = 0,
        LMS_ASSEMBLY_STATUS_WORKING,
        LMS_ASSEMBLY_STATUS_FINISHED,
        LMS_ASSEMBLY_STATUS_STATUS_MAX,
    };
    template <typename T>
    class SparseMatrix : public MatrixBase
    {
        static_assert(std::is_floating_point_v<T>, "SparseMatrix requires floating-point T");
        using Vec = std::vector<T>;

    public:
        SparseMatrix() = default;
        virtual ~SparseMatrix() = default;

        /**
         * @brief 重置矩阵为 colSize×colSize 方阵并清空旧数据
         * @param colSize 矩阵列数（方阵，即行数），必须大于 2
         * @return LMS_OK 成功；LMS_INI_ERR 尺寸非法（<= 2）
         * @note 按每行约 7 个非零元（如 3D CFD 7 点模板）预留 CSR 内存；
         *       装配状态置为 INVALID，之后以 AddRow(vec,row) / AddValue 逐行/逐元装配
         */
        unsigned short Reset(size_t colSize = 0) override
        {
            // 清理旧资源
            Clear();

            // 太小的数组没有意义
            if (colSize <= 2)
            {
                return LMS_INI_ERR;
            }

            // 列数固定
            isValid_ = true;
            assemblyStatus_ = LMS_ASSEMBLY_STATUS_INVALID;
            colSize_ = colSize;
            rowSize_ = colSize;
            value_.reserve(colSize * 7);
            colIdx_.reserve(colSize * 7);
            rowPtr_.reserve(colSize + 1);
            diagIdx_.reserve(colSize);
            cooTriples_.reserve(colSize * 21);
            // 初始行数偏移为0
            rowPtr_.push_back(0);
            return LMS_OK;
        }

        /**
         * @brief 清空矩阵所有数据，恢复初始无效状态（判零阈值同步复位）
         */
        void Clear() override
        {
            isValid_ = false;
            assemblyStatus_ = LMS_ASSEMBLY_STATUS_INVALID;
            colSize_ = 0;
            rowSize_ = 0;
            structureRevision_ = 0;
            eps_ = static_cast<T>(DEFAULT_EPSILON);
            value_.clear();
            colIdx_.clear();
            rowPtr_.clear();
            diagIdx_.clear();
            cooTriples_.clear();
        }

        /**
         * @brief 清空矩阵系数内容（保留矩阵尺寸，可重新填充）
         * @note 清空 CSR/COO 数据并重置行计数，colSize_/isValid_ 保留，
         *       清空后可按原方式重新组装矩阵
         */
        void ClearMat()
        {
            assemblyStatus_ = LMS_ASSEMBLY_STATUS_INVALID;
            structureRevision_++; // 结构变化，求解器需重染色

            value_.clear();
            colIdx_.clear();
            rowPtr_.clear();
            diagIdx_.clear();
            cooTriples_.clear();
            // 初始行数偏移为0
            rowPtr_.push_back(0);
        }

        void ZeroCoeff()
        {
            std::fill(value_.begin(), value_.end(), T{});
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
         * @brief 获取当前装配状态
         * @return assemblyStatus_
         */
        SparseMatrixAssemblyStatus AssemblyStatus() const
        {
            return assemblyStatus_;
        }

        /**
         * @brief 获取矩阵非零元素总数
         * @return 非零元个数（CSR value_ 数组长度）
         */
        size_t NotZeroElemNum() const
        {
            return value_.size();
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
         * @details 每次结构变化（ClearMat / AddRow）递增，用于求解器判断
         *          是否需重建红黑染色缓存等与结构相关的预处理数据
         * @return 当前结构版本号
         */
        size_t StructureRevision() const
        {
            return structureRevision_;
        }

    public:
        //  ====================== COO 动态装配 API ======================
        /**
         * @note COO 动态装配允许以任意 row 号添加矩阵行/元素：添加过程仅向三元组数组
         *       追加 (col, row, value)，不做结构约束；全部添加完成后调用 Finalize
         *       统一排序、压缩并转换为标准 CSR 存储结构。
         */

        /**
         * @brief 以 COO 方式添加矩阵的一行（稠密输入）
         * @param vec 该行数据，长度必须 >= colSize，仅前 colSize 个元素有效
         * @param row 目标行号（0 起始，须 < colSize），非零元以 (col, row) 三元组追加
         * @return LMS_OK 成功；
         *         LMS_STATUS_REJECT 矩阵已 Finalize（不可再装配）；
         *         LMS_INI_ERR 矩阵未初始化；
         *         LMS_OUT_SIZE_LIMIT 行号越界；
         *         LMS_VEC_LEN_ERR 向量长度不足；
         *         LMS_NAN_ELEM_ERR 元素含 NaN/Inf
         * @note 对角元校验延后到 Finalize 统一执行，允许先填非对角元、后补对角元；
         *       同一 (row,col) 可被多次添加，Finalize 时会自动累加合并
         */

        unsigned short AddRow(const Vec &vec, size_t row)
        {
            if (assemblyStatus_ == LMS_ASSEMBLY_STATUS_FINISHED)
            {
                return LMS_STATUS_REJECT;
            }
            auto result = AddRowCheck(vec, row);
            if (result != LMS_OK)
            {
                return result;
            }
            for (size_t i = 0; i < colSize_; i++)
            {
                if (IsZeroElem(vec[i], eps_))
                {
                    continue;
                }
                cooTriples_.push_back({i, row, vec[i]});
            }
            assemblyStatus_ = LMS_ASSEMBLY_STATUS_WORKING;
            return LMS_OK;
        }

        /**
         * @brief 以 COO 方式添加单个非零元（可重复，Finalize 时自动累加合并）
         * @param value 非零元值（|value| <= eps 视为零，跳过存储）
         * @param col 列号（须 < colSize）
         * @param row 行号（须 < colSize）
         * @return LMS_OK 成功；
         *         LMS_STATUS_REJECT 矩阵已 Finalize（不可再装配）；
         *         LMS_INI_ERR 矩阵未初始化；
         *         LMS_OUT_SIZE_LIMIT 行列越界；
         *         LMS_NAN_ELEM_ERR 值含 NaN/Inf
         */
        unsigned short AddValue(T value, size_t col, size_t row)
        {
            if (assemblyStatus_ == LMS_ASSEMBLY_STATUS_FINISHED)
            {
                return LMS_STATUS_REJECT;
            }
            auto result = AddValueCheck(value, col, row);
            if (result != LMS_OK)
            {
                return result;
            }
            cooTriples_.push_back({col, row, value});
            assemblyStatus_ = LMS_ASSEMBLY_STATUS_WORKING;
            return LMS_OK;
        }

        /**
         * @brief 查询指定 (col,row) 非零元在 value_ 中的下标（仅 Finalize 后可用）
         * @param idx 输出：非零元在 value_ 中的下标
         * @param col 列号
         * @param row 行号
         * @return LMS_OK 成功；LMS_STATUS_REJECT 未 Finalize；LMS_ERR 未找到该非零元
         */
        unsigned short ValueIdx(size_t &idx, size_t col, size_t row)
        {
            if (assemblyStatus_ != LMS_ASSEMBLY_STATUS_FINISHED)
            {
                return LMS_STATUS_REJECT;
            }
            if (rowPtr_.size() <= row)
            {
                return LMS_ERR;
            }

            for (size_t i = rowPtr_[row]; i < rowPtr_[row + 1]; i++)
            {
                if (colIdx_[i] == col)
                {
                    idx = i;
                    return LMS_OK;
                }
            }
            return LMS_ERR;
        }

        /**
         * @brief 覆盖指定 (col,row) 非零元的值（仅 Finalize 后可用，该非零元须已存在）
         * @param value 新值
         * @param col 列号
         * @param row 行号
         * @return LMS_OK 成功；LMS_STATUS_REJECT 未 Finalize；
         *         LMS_INI_ERR 未初始化；LMS_OUT_SIZE_LIMIT 行列越界；
         *         LMS_NAN_ELEM_ERR 值含 NaN/Inf；LMS_ERR 未找到该非零元
         */
        unsigned short SetValue(T value, size_t col, size_t row)
        {
            if (assemblyStatus_ != LMS_ASSEMBLY_STATUS_FINISHED)
            {
                return LMS_STATUS_REJECT;
            }
            auto result = AddValueCheck(value, col, row);
            if (result != LMS_OK)
            {
                return result;
            }
            size_t idx = 0;
            result = ValueIdx(idx, col, row);
            if (result == LMS_OK)
            {
                value_[idx] = value;
            }
            return result;
        }

        /**
         * @brief 读取指定 (col,row) 的非零元值（仅 Finalize 后可用）
         * @param value 输出：该位置的非零元值
         * @param col 列号
         * @param row 行号
         * @return LMS_OK 成功；LMS_STATUS_REJECT 状态非法；
         *         LMS_OUT_SIZE_LIMIT 行列越界；LMS_ERR 未找到该非零元
         */
        unsigned short GetValue(T &value, size_t col, size_t row)
        {
            if ((colSize_ == 0) || (rowSize_ == 0) || (!isValid_) ||
                assemblyStatus_ != LMS_ASSEMBLY_STATUS_FINISHED)
            {
                return LMS_STATUS_REJECT;
            }
            if ((col >= colSize_) || (row >= rowSize_))
            {
                return LMS_OUT_SIZE_LIMIT;
            }
            size_t idx = 0;
            auto result = ValueIdx(idx, col, row);
            if (result == LMS_OK)
            {
                value = value_[idx];
            }
            return result;
        }

        /**
         * @brief COO 装配收尾：排序、压缩并转换为标准 CSR 结构
         * @details ① 将三元组按 (row, col) 升序排序；
         *          ② 单趟压缩：合并重复 (row,col) 的值，生成 value_/colIdx_/rowPtr_；
         *          ③ 定位每行对角元 diagIdx_ 并校验（行缺对角元或对角元为零则失败）；
         *          ④ 成功后清空 COO 数组，装配状态置为 FINISHED，矩阵可求解
         * @return LMS_OK 成功；
         *         LMS_STATUS_REJECT 当前状态非 WORKING（未装配或已收尾）；
         *         LMS_COOTRIPLES_EMPTY COO 数组为空；
         *         LMS_DIAG_ELEM_ZERO 存在行缺少对角元或对角元为零
         */
        unsigned short Finalize()
        {
            if (assemblyStatus_ != LMS_ASSEMBLY_STATUS_WORKING)
            {
                return LMS_STATUS_REJECT;
            }
            if (cooTriples_.empty())
            {
                return LMS_COOTRIPLES_EMPTY;
            }
            std::sort(cooTriples_.begin(), cooTriples_.end(),
                      [](const CooTriples &a, const CooTriples &b)
                      {
                          return (a.row == b.row) ? (a.col < b.col) : (a.row < b.row);
                      });

            value_.clear();
            colIdx_.clear();
            rowPtr_.assign(colSize_ + 1, 0);
            diagIdx_.assign(colSize_, SIZE_MAX);

            size_t k = 0;
            const size_t n = cooTriples_.size();
            for (size_t row = 0; row < colSize_; ++row) // 逐行，天然覆盖空行
            {
                rowPtr_[row] = value_.size();           // 本行起始（空行也正确）
                if (k < n && cooTriples_[k].row == row) // 该行有元素
                {
                    while (k < n && cooTriples_[k].row == row)
                    {
                        size_t col = cooTriples_[k].col;
                        T sum = 0;
                        while (k < n && cooTriples_[k].row == row &&
                               cooTriples_[k].col == col) // 合并同 (row,col)
                        {
                            sum += cooTriples_[k].value;
                            ++k;
                        }
                        value_.push_back(sum);
                        colIdx_.push_back(col);
                        if (col == row)
                        {
                            diagIdx_[row] = colIdx_.size() - 1;
                        }
                    }
                }
            }
            rowPtr_[colSize_] = value_.size(); // 收尾：总非零元数

            // 对角元校验
            for (size_t i = 0; i < colSize_; i++)
            {
                for (size_t j = rowPtr_[i]; j < rowPtr_[i + 1]; j++)
                {
                    if (colIdx_[j] == i)
                    {
                        diagIdx_[i] = j;
                        break;
                    }
                }
                if ((diagIdx_[i] == SIZE_MAX) || IsZeroElem(value_[diagIdx_[i]], eps_))
                {
                    value_.clear();
                    colIdx_.clear();
                    rowPtr_.clear();
                    diagIdx_.clear();
                    return LMS_DIAG_ELEM_ZERO;
                }
            }

            assemblyStatus_ = LMS_ASSEMBLY_STATUS_FINISHED;
            structureRevision_++;
            cooTriples_.clear();
            return LMS_OK;
        }

        //  ====================== 内部辅助（校验 / 私有成员） ======================
    private:
        /**
         * @brief 检查矩阵是否严格对角占优（|Aii| > Σ|Aij|，j≠i）
         * @return LMS_DIAG_DOM 严格占优；
         *         LMS_NOT_DIAG_DOM 不占优；
         *         LMS_INI_ERR/LMS_SPARSE_ROW_NOT_FULL 状态非法
         * @note 严格对角占优是 SOR 收敛的充分条件而非必要条件，
         *       不占优不代表发散，仅提示风险
         */
        unsigned short IsStrictlyDiagonallyDominant() const
        {
            auto result = Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_);
                 },
                 LMS_INI_ERR},
                {[&]() -> bool
                 {
                     return this->rowSize_ < this->colSize_;
                 },
                 LMS_SPARSE_ROW_NOT_FULL},
            });
            if (result != LMS_OK)
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
                    return LMS_NOT_DIAG_DOM;
                }
            }
            return LMS_DIAG_DOM;
        }

        /**
         * @brief 添加矩阵的一行前的异常检查（DEFAULT 与 COO 共用）
         * @param vec 待添加的行向量
         * @param row 目标行号（用于越界检查）
         * @return LMS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short AddRowCheck(const Vec &vec, size_t row)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (this->rowSize_ == 0) || (!this->isValid_);
                 },
                 LMS_INI_ERR},
                {[&]() -> bool
                 {
                     return row >= this->rowSize_;
                 },
                 LMS_OUT_SIZE_LIMIT},
                {[&]() -> bool
                 {
                     return vec.size() < this->colSize_;
                 },
                 LMS_VEC_LEN_ERR},
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
                 LMS_NAN_ELEM_ERR},
            });
        }

        unsigned short AddValueCheck(T value, size_t col, size_t row)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (this->rowSize_ == 0) || (!this->isValid_);
                 },
                 LMS_INI_ERR},
                {[&]() -> bool
                 {
                     return (col >= this->colSize_) || (row >= this->rowSize_);
                 },
                 LMS_OUT_SIZE_LIMIT},
                {[&]() -> bool
                 {
                     return (std::isnan(value) || std::isinf(value));
                 },
                 LMS_NAN_ELEM_ERR},
            });
        }

    private:
        bool isValid_ = {false};                                                    // 是否已初始化（Reset 成功）
        SparseMatrixAssemblyStatus assemblyStatus_ = {LMS_ASSEMBLY_STATUS_INVALID}; // 装配状态
        size_t colSize_ = {0};                                                      // 矩阵列数
        size_t rowSize_ = {0};                                                      // 矩阵行数
        size_t structureRevision_ = {0};                                            // 矩阵结构版本号（结构变化时递增）
        T eps_ = static_cast<T>(DEFAULT_EPSILON);                                   // 判零阈值
        std::vector<T> value_;                                                      // 非零元素值（CSR）
        std::vector<size_t> colIdx_;                                                // 非零元素列索引（CSR）
        std::vector<size_t> rowPtr_;                                                // 每行起始偏移（CSR，长度 = 行数+1）
        std::vector<size_t> diagIdx_;                                               // 每行对角元在 value_ 中的下标
        // COO 动态添加矩阵行相关参数
        /// @brief COO 坐标三元组：一个非零元的列、行与值
        struct CooTriples
        {
            size_t col; // 列号
            size_t row; // 行号
            T value;    // 值
        };
        std::vector<CooTriples> cooTriples_; // COO 三元组数组（动态装配暂存）
    };
}

#endif // LMS_SPARSE_MATRIX_HPP
