/**
 * @file light_matrix_solver.hpp
 * @brief 轻量级稀疏矩阵线性求解头文件库（header-only，C++17）
 * @details 基于 CSR（Compressed Sparse Row，压缩稀疏行）稀疏存储的线性方程组求解库，
 *          适用于 CFD 等大规模稀疏线性系统。
 *
 *          采用「数据 / 算法解耦 + 策略模式」的可插拔架构：
 *          - SparseMatrix：矩阵数据容器，只负责存储与组装（CSR 三元组 + 对角元索引），不含算法；
 *          - LinearSolverBase：求解算法抽象基类，定义统一求解接口
 *            （Solve / 求解模式 / 收敛状态），供上层框架按基类指针运行时可切换求解器；
 *          - 具体求解器继承基类并实现 Solve：当前实现 SOR（逐次超松弛）与
 *            红黑染色并行 SOR，未来可扩展 Krylov 子空间法（CG / BiCGSTAB / GMRES）、
 *            多重网格（AMG / GAMG）等，上层调用无需改动。
 *
 *          用法：
 *          - 组装矩阵：Reset → AddRow / AddMultiRow / AddMat；
 *          - 重填矩阵：ClearMat → AddRow / AddMat（保留矩阵尺寸）；
 *          - 求解：solver.Solve(mat, x, rhs)（右端项由调用方持有）。
 */
#ifndef LIGHT_MATRIX_SOLVER_H
#define LIGHT_MATRIX_SOLVER_H

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
        LMS_MAT_SIZE_ERR,        // 矩阵大小不足
        LMS_DIAG_ELEM_ZERO,      // 对角元素为零（SOR 无法迭代）
        LMS_NAN_ELEM_ERR,        // 输入含 NaN/Inf
        LMS_SPARSE_ROW_FULL,     // 矩阵行已满，无法继续添加
        LMS_SPARSE_ROW_NOT_FULL, // 矩阵未装满，无法求解
        LMS_SOLVE_SET_ERR,       // 求解参数非法（iter/alphaW/tol 越界）
        LMS_NOT_DIAG_DOM,        // 不满足严格对角占优
        LMS_DIAG_DOM,            // 满足严格对角占优
        LMS_NOT_CONVERGED,       // 迭代跑满仍未收敛（x 不可信）
        LMS_DIVERGED,            // 迭代发散（残差爆炸 / NaN）
        LMS_MODE_ERR,            // 求解模式非法
    };

    /**
     * @brief 计算模式枚举
     * @note  串行模式（默认）/ 并行模式（红黑染色）/ 自动模式（规模与线程足够时并行）
     */
    enum CalcProcessMode : unsigned short
    {
        LMS_SERIAL_MODE = 0,    // 串行模式
        LMS_MULTI_PROCESS_MODE, // 并行模式（需 -fopenmp）
        LMS_AUTO_MODE,          // 自动模式（规模足够大且多线程时并行）
        LMS_MODE_MAX,
    };

    /**
     * @brief 求解器类型
     */
    enum LightMatrixSolverType : unsigned short
    {
        LMS_DEFAULT = 0,
        LMS_SOR_SOLVER,
        LMS_SOLVER_TYPE_MAX,
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
    struct CheckCondition
    {
        CheckFunc checkFunc = {nullptr};
        ErrCode errCode = {LMS_OK};
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
        size_t colSize_ = {0};   // 矩阵列数
        size_t rowSize_ = {0};   // 当前已添加的行数
        bool isValid_ = {false}; // 是否已初始化（Reset 成功）
    };

    // ====================== CSR存储数据矩阵 ======================
    /**
     * @brief 基于 CSR 稀疏存储的方阵类（数据容器，仅负责存储与组装，不含迭代算法）
     * @tparam T 浮点类型（float/double），编译期 static_assert 校验
     * @note 非零元素按行压缩存储：value_/colIdx_/rowPtr_ 构成标准 CSR 三元组，
     *       diagIdx_ 记录每行对角元在 value_ 中的下标（求解时 O(1) 访问）
     * @note 求解算法见 LinearSolverBase / SORSolver；矩阵通过只读访问器
     *       （Values/ColIdx/RowPtr/DiagIdx/Eps/StructureRevision）向求解器暴露内部 CSR 结构；
     *       右端项 b 不存入矩阵，由调用方以 Solve(mat, x, rhs) 参数传入。
     */
    template <typename T>
    class SparseMatrix : public MatrixBase
    {
        static_assert(std::is_floating_point_v<T>, "SparseMatrix requires floating-point T");
        using Arr = T *;
        using Vec = std::vector<T>;
        using Mat = std::vector<std::vector<T>>;

    public:
        SparseMatrix() = default;
        virtual ~SparseMatrix() = default;

        /**
         * @brief 重置矩阵为 colSize×colSize 方阵并清空旧数据
         * @param colSize 矩阵列数（方阵，即行数），必须大于 2
         * @return LMS_OK 成功；LMS_INI_ERR 尺寸非法（<= 2）
         * @note 按每行约 7 个非零元（如 3D CFD 7 点模板）预留内存
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
            colSize_ = colSize;
            isValid_ = true;
            value_.reserve(colSize * 7);
            colIdx_.reserve(colSize * 7);
            rowPtr_.reserve(colSize + 1);
            diagIdx_.reserve(colSize);

            // 初始行数偏移为0
            rowPtr_.push_back(0);

            return LMS_OK;
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
            structureRevision_ = 0;
        }

        /**
         * @brief 清空矩阵系数内容（保留矩阵尺寸，可重新填充）
         * @note 仅清空 CSR 数据并重置行计数，colSize_/isValid_ 保留；
         *       清空后可再次 AddRow/AddMat 重新组装矩阵
         */
        void ClearMat()
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

        /**
         * @brief 添加矩阵的一行（稠密输入，内部压缩为 CSR 存储）
         * @param vec 该行数据，长度必须 >= colSize，仅前 colSize 个元素有效
         * @return LMS_OK 成功；
         *         LMS_INI_ERR 矩阵未初始化；LMS_SPARSE_ROW_FULL 行已满；
         *         LMS_VEC_LEN_ERR 向量长度不足；
         *         LMS_DIAG_ELEM_ZERO 对角元为零；
         *         LMS_NAN_ELEM_ERR 元素含 NaN/Inf
         * @note 每行对角元（vec[rowSize_]）必须非零，否则 SOR 无法迭代；
         *       NaN/Inf 检查通过后才写入，失败不会留下部分数据
         */
        unsigned short AddRow(const Vec &vec)
        {
            auto result = AddRowByVecAbnormalCheck(vec);
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
            return LMS_OK;
        }

        /**
         * @brief 追加批量行（等价于连续调用 AddRow，保留已有行）
         * @param mat 待追加的稠密行数组（行向量数组）
         * @param errRow 输出参数：失败时返回出错行的行号（0 起始）
         * @return 同 AddRow
         * @note 与 AddMat 的区别：AddMat 会先清空矩阵再整体重建（覆盖语义），
         *       本函数在现有行之后继续追加；任一行失败时矩阵被 ClearMat
         *       （保留矩阵尺寸，可修正后重试）
         */
        unsigned short AddMultiRow(const Mat &mat, size_t &errRow)
        {
            unsigned short result = LMS_OK;
            for (size_t i = 0; i < mat.size(); i++)
            {
                result = AddRow(mat[i]);
                if (result != LMS_OK)
                {
                    errRow = i;
                    ClearMat();
                    return result;
                }
            }
            return LMS_OK;
        }

        /**
         * @brief 覆盖式批量组装矩阵（先清空现有行，再整体添加）
         * @param mat 稠密矩阵（行向量数组），行数不能少于 colSize（超出的行按满行报错）
         * @param errRow 输出参数：失败时返回出错行的行号（0 起始）
         * @return LMS_MAT_SIZE_ERR 行数不足 colSize；其余同 AddRow
         * @note 开头先 ClearMat 清空已有行（保留矩阵尺寸）；
         *       任一行失败时矩阵被 ClearMat（保留矩阵尺寸，可修正后重试）
         */
        unsigned short AddMat(const Mat &mat, size_t &errRow)
        {
            ClearMat();
            // 检查是否能添加满
            if (mat.size() < colSize_)
            {
                return LMS_MAT_SIZE_ERR;
            }
            unsigned short result = LMS_OK;
            for (size_t i = 0; i < mat.size(); i++)
            {
                result = AddRow(mat[i]);
                if (result != LMS_OK)
                {
                    errRow = i;
                    ClearMat();
                    return result;
                }
            }
            return LMS_OK;
        }

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

    private:
        /**
         * @brief 添加矩阵的一行前的异常检查
         * @param vec 待添加的行向量
         * @return LMS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short AddRowByVecAbnormalCheck(const Vec &vec)
        {
            return Check({
                {[&]() -> bool
                 {
                     return (this->colSize_ == 0) || (!this->isValid_);
                 },
                 LMS_INI_ERR},
                {[&]() -> bool
                 {
                     return this->rowSize_ >= this->colSize_;
                 },
                 LMS_SPARSE_ROW_FULL},
                {[&]() -> bool
                 {
                     return vec.size() < this->colSize_;
                 },
                 LMS_VEC_LEN_ERR},
                {[&]() -> bool
                 {
                     return IsZeroElem(vec[this->rowSize_], this->eps_);
                 },
                 LMS_DIAG_ELEM_ZERO},
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

    private:
        std::vector<T> value_;                    // 非零元素值（CSR）
        std::vector<size_t> colIdx_;              // 非零元素列索引（CSR）
        std::vector<size_t> rowPtr_;              // 每行起始偏移（CSR，长度 = 行数+1）
        std::vector<size_t> diagIdx_;             // 每行对角元在 value_ 中的下标
        T eps_ = static_cast<T>(DEFAULT_EPSILON); // 判零阈值
        size_t structureRevision_ = {0};          // 矩阵结构版本号（结构变化时递增）
    };

    // ====================== 求解器基类 ======================
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
        using Mat = std::vector<std::vector<T>>;

    public:
        virtual ~LinearSolverBase() = default;
        LinearSolverBase(const LinearSolverBase &) = delete;
        LinearSolverBase(LinearSolverBase &&) = delete;
        LinearSolverBase &operator=(const LinearSolverBase &) = delete;
        LinearSolverBase &operator=(LinearSolverBase &&) = delete;

        /**
         * @brief 求解线性方程组 A x = b（A 由矩阵 mat 提供）
         * @param mat 系数矩阵（数据容器，求解器只读）
         * @param x 输入输出参数：进入时为零初值或上次解（作初值猜测），
         *          退出时为解向量（长度必须 >= mat.ColSize()）
         * @param rhs 右端项
         * @return LMS_OK 收敛；其余为错误码（见 ErrCode）
         */
        virtual unsigned short Solve(const SparseMatrix<T> &mat, Vec &x, const Vec &rhs) = 0;

        /// @brief 查询求解器类型
        virtual LightMatrixSolverType GetSolverType() const = 0;

        /// @brief 获取最后一次求解的最终残差（未求解时为 0）
        virtual T GetResidual() const
        {
            return finalRes_;
        }

        /// @brief 获取最后一次求解实际执行的迭代次数
        virtual size_t GetIterCount() const
        {
            return iterCnt_;
        }

        /// @brief 查询最后一次求解是否收敛
        virtual bool IsConverged() const
        {
            return converged_;
        }

        /// @brief 设置求解模式 串行/并行
        virtual bool SetCalcProcessMode(unsigned short mode)
        {
            if (mode >= static_cast<unsigned short>(LMS_MODE_MAX))
            {
                return false;
            }
            mode_ = static_cast<CalcProcessMode>(mode);
            return true;
        }

        virtual bool SetCalcProcessMode(CalcProcessMode mode)
        {
            if (mode >= LMS_MODE_MAX)
            {
                return false;
            }
            mode_ = mode;
            return true;
        }
        /// @brief 查询当前求解模式
        virtual CalcProcessMode GetCalcProcessMode() const
        {
            return mode_;
        }

    protected:
        LinearSolverBase() = default;

        T finalRes_ = T{};         // 最终残差
        size_t iterCnt_ = {0};     // 实际迭代次数
        bool converged_ = {false}; // 是否收敛

        CalcProcessMode mode_ = {LMS_SERIAL_MODE}; // 计算模式
    };

    // ====================== SOR求解器实现 ======================
    /**
     * @brief SOR（逐次超松弛）求解器实现
     * @tparam T 浮点类型（float/double）
     * @details 提供串行与红黑染色并行两种求解模式，通过 SetCalcProcessMode 切换：
     *          - SOR_SERIAL_MODE（默认）：经典逐行 Gauss-Seidel/SOR 迭代；
     *          - SOR_MULTI_PROCESS_MODE：红黑染色并行 SOR（需 -fopenmp）。
     *            将未知量按矩阵拓扑二分（红/黑），每轮迭代拆成两个并行 pass——
     *            红点只读黑点旧值、黑点读红点新值，隐式 barrier 同步，保留顺序
     *            收敛语义；非二分图（含奇环）自动退化为串行。
     *          - SOR_AUTO_MODE：规模足够大且多线程时自动启用并行，否则走串行。
     */
    template <typename T>
    class SORSolver : public LinearSolverBase<T>
    {
        using Arr = T *;
        using Vec = std::vector<T>;
        using Mat = std::vector<std::vector<T>>;

    public:
        /**
         * @brief 构造 SOR 求解器
         * @param maxIter 最大迭代次数，合法范围 [MIN_ITER, MAX_ITER]
         * @param omega 松弛因子（1.0 即 Gauss-Seidel），合法范围 (0, MAX_ALPHAW]
         * @param tol 收敛容差（基于归一化残差 ||Ax-b||/||b||），合法范围 (0, MAX_SET_TOL]
         * @note 默认串行模式；并行 / 自动模式通过 SetCalcProcessMode 切换
         */
        SORSolver(size_t maxIter, T omega, T tol = static_cast<T>(DEFAULT_TOL))
            : maxIter_(maxIter), omega_(omega), tol_(tol)
        {
        }

        /**
         * @brief 求解线性方程组 A x = b
         * @param mat 系数矩阵（数据容器，求解器只读）
         * @param x 输入输出参数：进入时为零初值或上次解（作初值猜测），
         *          退出时为解向量（长度必须 >= mat.ColSize()）
         * @param rhs 右端项
         *         LMS_INI_ERR 矩阵未初始化；
         *         LMS_VEC_LEN_ERR x长度不足；
         *         LMS_SPARSE_ROW_NOT_FULL 矩阵未填充满至方阵；
         *         LMS_SOLVE_SET_ERR 构造参数 iter/omega/tol 非法；
         *         LMS_NOT_CONVERGED 迭代跑满未达标（x 为最后迭代值，不可信）；
         *         LMS_DIVERGED 发散（NaN/Inf 或残差爆炸）；
         *         其余为参数/状态校验错误码
         * @note 每 10 次迭代检测一次残差；收敛或发散时提前返回，x 保留当时结果；
         *       计算模式由 SetCalcProcessMode 决定（串行 / 红黑并行 / 自动）
         */
        unsigned short Solve(const SparseMatrix<T> &mat, Vec &x, const Vec &rhs) override
        {
            auto result = SolverParaCheck(mat, x, rhs);
            if (result != LMS_OK)
            {
                return result;
            }

#ifdef _OPENMP
            maxThread_ = static_cast<unsigned short>(omp_get_max_threads());
#endif
            switch (this->mode_)
            {
            case LMS_SERIAL_MODE:
            {
                result = SerialSolve(mat, x, rhs);
                break;
            }
            case LMS_MULTI_PROCESS_MODE:
            {
                result = MultiProcSolve(mat, x, rhs);
                break;
            }
            case LMS_AUTO_MODE:
            {
                if ((maxThread_ == 1) || (mat.ColSize() / maxThread_ < 1000))
                {
                    result = SerialSolve(mat, x, rhs);
                }
                else
                {
                    result = MultiProcSolve(mat, x, rhs);
                }
                break;
            }
            default:
            {
                result = static_cast<decltype(result)>(LMS_MODE_ERR);
            }
            }

            return result;
        }

        LightMatrixSolverType GetSolverType() const override
        {
            return LMS_SOR_SOLVER;
        }

    private:
        /**
         * @brief 求解前参数与状态校验
         * @param x 解向量
         * @param mat 系数矩阵
         * @return LMS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short SolverParaCheck(const SparseMatrix<T> &mat, const Vec &x, const Vec &rhs) const
        {
            return Check({
                {[&]() -> bool
                 {
                     return (!mat.IsValid()) ||
                            (mat.ColSize() == 0);
                 },
                 LMS_INI_ERR},
                {[&]() -> bool
                 {
                     return (x.size() < mat.ColSize()) ||
                            (rhs.size() < mat.ColSize());
                 },
                 LMS_VEC_LEN_ERR},
                {[&]() -> bool
                 {
                     for (size_t i = 0; i < mat.ColSize(); ++i)
                     {
                         if (std::isnan(x[i]) || std::isinf(x[i]) ||
                             std::isnan(rhs[i]) || std::isinf(rhs[i]))
                         {
                             return true;
                         }
                     }
                     return false;
                 },
                 LMS_NAN_ELEM_ERR},
                {[&]() -> bool
                 {
                     return mat.RowSize() < mat.ColSize();
                 },
                 LMS_SPARSE_ROW_NOT_FULL},
                {[&]() -> bool
                 {
                     return ((this->maxIter_ < MIN_ITER) ||
                             (this->maxIter_ > MAX_ITER) ||
                             ((this->omega_ <= 0) || (this->omega_ > static_cast<T>(MAX_ALPHAW))) ||
                             ((this->tol_ <= 0) || (this->tol_ > static_cast<T>(MAX_SET_TOL))));
                 },
                 LMS_SOLVE_SET_ERR},
            });
        }

        /**
         * @brief 迭代过程中的发散检查
         * @param res 当前残差
         * @param res0 初始残差
         * @param eps 判零阈值
         * @return LMS_OK 正常；LMS_DIVERGED 发散
         */
        unsigned short DivergenceCheck(T res, T res0, T eps) const
        {
            return Check({
                {[&]() -> bool
                 {
                     return std::isnan(res) || std::isinf(res) ||
                            (res > static_cast<T>(MAX_OVERLIMIT_TOL)) ||
                            ((res0 > eps) && (res > 1000 * res0));
                 },
                 LMS_DIVERGED},
            });
        }

        /**
         * @brief 串行 SOR 迭代
         * @param x 解向量（原地更新）
         * @param mat 系数矩阵
         * @param bNorm 均匀残差 ||b||₂
         * @param res0 初始残差
         * @return 收敛/发散/未收敛错误码
         */
        unsigned short SerialSolve(const SparseMatrix<T> &mat, Vec &x, const Vec &rhs)
        {
            const auto &value = mat.Values();
            const auto &colIdx = mat.ColIdx();
            const auto &rowPtr = mat.RowPtr();
            const auto &diagIdx = mat.DiagIdx();
            const size_t n = mat.ColSize();
            const T eps = mat.Eps();

            this->iterCnt_ = 0;
            this->converged_ = false;
            const size_t checkEvery = 10;

            // 归一化残差 / 初始残差
            T bNorm = ComputeUniformResidual(mat, rhs);
            T res0 = ComputeResidual(mat, x, rhs);

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
                    T res = ComputeResidual(mat, x, rhs);
                    this->finalRes_ = res;
                    this->iterCnt_ = it + 1;
                    auto err = DivergenceCheck(res, res0, eps);
                    if (err != LMS_OK)
                    {
                        return err;
                    }
                    // 归一化残差收敛判断
                    T resNorm = IsZeroElem(bNorm, eps) ? res : (res / bNorm);
                    if (resNorm < tol_)
                    {
                        this->converged_ = true;
                        return LMS_OK;
                    }
                }
            }
            return LMS_NOT_CONVERGED;
        }

        /**
         * @brief 并行 SOR 迭代
         * @param x 解向量（原地更新）
         * @param mat 系数矩阵
         * @param bNorm 均匀残差 ||b||₂
         * @param res0 初始残差
         * @return 收敛/发散/未收敛错误码
         */
        unsigned short MultiProcSolve(const SparseMatrix<T> &mat, Vec &x, const Vec &rhs)
        {
            if ((maxThread_ == 1) || !BuildRedBlackColoring(mat))
            {
                return SerialSolve(mat, x, rhs); // 非二分图退化串行
            }

            const auto &value = mat.Values();
            const auto &colIdx = mat.ColIdx();
            const auto &rowPtr = mat.RowPtr();
            const auto &diagIdx = mat.DiagIdx();
            const T eps = mat.Eps();

            this->iterCnt_ = 0;
            this->converged_ = false;
            const size_t checkEvery = 10;
            // 归一化残差 / 初始残差
            T bNorm = ComputeUniformResidual(mat, rhs);
            T res0 = ComputeResidual(mat, x, rhs);

            for (size_t it = 0; it < maxIter_; it++)
            {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < redList_.size(); i++)
                {
                    size_t j = redList_[i];
                    T sum = rhs[j];
                    for (size_t k = rowPtr[j]; k < rowPtr[j + 1]; k++)
                    {
                        if (colIdx[k] != j)
                        {
                            sum -= value[k] * x[colIdx[k]];
                        }
                    }
                    x[j] = (1 - omega_) * x[j] + omega_ * sum / value[diagIdx[j]];
                }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < blackList_.size(); i++)
                {
                    size_t j = blackList_[i];
                    T sum = rhs[j];
                    for (size_t k = rowPtr[j]; k < rowPtr[j + 1]; k++)
                    {
                        if (colIdx[k] != j)
                        {
                            sum -= value[k] * x[colIdx[k]];
                        }
                    }
                    x[j] = (1 - omega_) * x[j] + omega_ * sum / value[diagIdx[j]];
                }

                if ((it + 1) % checkEvery == 0)
                {
                    T res = ComputeResidual(mat, x, rhs);
                    this->finalRes_ = res;
                    this->iterCnt_ = it + 1;
                    auto err = DivergenceCheck(res, res0, eps);
                    if (err != LMS_OK)
                    {
                        return err;
                    }
                    // 归一化残差收敛判断
                    T resNorm = IsZeroElem(bNorm, eps) ? res : (res / bNorm);
                    if (resNorm < tol_)
                    {
                        this->converged_ = true;
                        return LMS_OK;
                    }
                }
            }
            return LMS_NOT_CONVERGED;
        }

        /**
         * @brief 计算右端项均匀范数 ||b||₂（归一化残差的分母）
         * @param mat 系数矩阵（用于取列数，保证与残差口径一致）
         * @param rhs 右端项
         * @return ||b||₂ = sqrt(Σ_{i<ColSize} rhs[i]²)
         */
        T ComputeUniformResidual(const SparseMatrix<T> &mat, const Vec &rhs) const
        {
            T res = T{};
            for (size_t i = 0; i < mat.ColSize(); i++)
            {
                res += rhs[i] * rhs[i];
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
        T ComputeResidual(const SparseMatrix<T> &mat, const Vec &x, const Vec &rhs) const
        {
            const auto &value = mat.Values();
            const auto &colIdx = mat.ColIdx();
            const auto &rowPtr = mat.RowPtr();
            T res = T{};
            for (size_t i = 0; i < mat.ColSize(); i++)
            {
                size_t startPos = rowPtr[i];
                size_t endPos = rowPtr[i + 1];
                T rhs_i = rhs[i];
                for (size_t j = startPos; j < endPos; j++)
                {
                    rhs_i -= value[j] * x[colIdx[j]];
                }
                res += rhs_i * rhs_i;
            }
            return std::sqrt(res);
        }

        /// @return true=染色成功（二分图）；false=含奇环，非二分图（调用方退化为串行）
        bool BuildRedBlackColoring(const SparseMatrix<T> &mat)
        {
            if ((mat.StructureRevision() == coloredRevision_) &&
                (color_.size() == mat.ColSize()))
            {
                return true; // 结构未变，复用缓存染色
            }

            const size_t n = mat.ColSize();
            const auto &rowPtr = mat.RowPtr();
            const auto &colIdx = mat.ColIdx();

            std::vector<unsigned char> color(n, 0); // 0未染色，1-红色，2-黑色
            std::vector<size_t> queue;

            for (size_t i = 0; i < n; i++)
            {
                if (color[i])
                {
                    continue;
                }
                color[i] = 1;
                queue.clear();
                queue.push_back(i);
                size_t head = 0;
                while (head < queue.size())
                {
                    size_t u = queue[head++];
                    for (size_t j = rowPtr[u]; j < rowPtr[u + 1]; j++)
                    {
                        if (colIdx[j] == u)
                        {
                            continue;
                        }
                        if (!color[colIdx[j]])
                        {
                            color[colIdx[j]] = (color[u] == 1) ? 2 : 1;
                            queue.push_back(colIdx[j]);
                        }
                        else if (color[colIdx[j]] == color[u])
                        {
                            return false;
                        }
                    }
                }
            }
            redList_.clear();
            blackList_.clear();
            for (size_t i = 0; i < n; i++)
            {
                if (color[i] == 1)
                {
                    redList_.push_back(i);
                }
                else
                {
                    blackList_.push_back(i);
                }
            }
            color_ = std::move(color);
            coloredRevision_ = mat.StructureRevision();
            return true;
        }

    private:
        size_t maxIter_; // 最大迭代次数
        T omega_;        // 松弛因子
        T tol_;          // 收敛容差

        // 并行SOR相关变量
        unsigned short maxThread_ = {1};   // 最大线程数
        std::vector<unsigned char> color_; // 染色结果：1=红，2=黑
        std::vector<size_t> redList_;      // 红点列表
        std::vector<size_t> blackList_;    // 黑点列表
        size_t coloredRevision_;           // 已染色的矩阵结构版本号
    };
}

#endif // LIGHT_MATRIX_SOLVER_H
