/**
 * @file light_matrix_solver.hpp
 * @brief 轻量级稀疏矩阵线性求解头文件库（header-only，C++17）
 * @details 基于 CSR（Compressed Sparse Row，压缩稀疏行）稀疏存储的线性方程组求解库，
 *          适用于 CFD 等大规模稀疏线性系统。
 *
 *          采用「数据 / 算法解耦 + 策略模式」的可插拔架构：
 *          - SparseMatrix（见 csr_matrix.hpp）：矩阵数据容器，只负责存储与组装，不含算法。
 *            支持两种装配方式：DEFAULT 顺序填充（AddRow / AddMultiRow / AddMat）与
 *            COO 动态装配（AddRow(vec,row) / Finalize），最终统一产出标准 CSR 结构；
 *          - LinearSolverBase：求解算法抽象基类，定义统一求解接口
 *            （Solve / 求解模式 / 收敛状态），供上层框架按基类指针运行时可切换求解器；
 *          - 具体求解器继承基类并实现 Solve：当前实现 SOR（逐次超松弛，含红黑染色并行）
 *            与 Jacobi（雅可比，含直接并行），未来可扩展 Krylov 子空间法
 *            （CG / BiCGSTAB / GMRES）、多重网格（AMG / GAMG）等，上层调用无需改动。
 *
 *          用法：
 *          - 顺序填充：Reset → AddRow / AddMultiRow / AddMat；
 *          - COO 动态装配：Reset → SetAssemblyMethod(COO) → AddRow(vec,row) → Finalize；
 *          - 重填矩阵：ClearMat → 重新组装（保留矩阵尺寸）；
 *          - 求解：solver.Solve(mat, x, rhs)（右端项由调用方持有）。
 */
#ifndef LIGHT_MATRIX_SOLVER_H
#define LIGHT_MATRIX_SOLVER_H

#include "csr_matrix.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace LMS
{
    /// @brief 求解器类型
    enum LightMatrixSolverType : unsigned short
    {
        LMS_SOLVER_TYPE_DEFAULT = 0, // 默认（未指定具体类型）
        LMS_SOLVER_TYPE_SOR,         // SOR（逐次超松弛）求解器
        LMS_SOLVER_TYPE_JACOBI,      // Jacobi（雅可比）求解器
        LMS_SOLVER_TYPE_MAX,
    };

    /**
     * @brief 计算模式枚举
     * @note  串行模式（默认）/ 并行模式（红黑染色 / 直接并行）/ 自动模式（规模与线程足够时并行）
     */
    enum CalcProcessMode : unsigned short
    {
        LMS_CALC_MODE_SERIAL = 0,         // 串行模式
        LMS_CALC_MODE_MULTI_PROCESS_MODE, // 并行模式（需 -fopenmp）
        LMS_CALC_MODE_AUTO,               // 自动模式（规模足够大且多线程时并行）
        LMS_CALC_MODE_MAX,
    };

    // ====================== 求解器基类 ======================
    /**
     * @brief 线性求解器抽象基类（策略模式）
     * @tparam T 浮点类型（float/double）
     * @details 定义统一求解接口，将"求解算法"与"矩阵数据"解耦。
     *          具体求解器（SOR / Jacobi / 未来的 GAMG / PCG 等）继承本类并实现 Solve。
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
            if (mode >= static_cast<unsigned short>(LMS_CALC_MODE_MAX))
            {
                return false;
            }
            mode_ = static_cast<CalcProcessMode>(mode);
            return true;
        }

        virtual bool SetCalcProcessMode(CalcProcessMode mode)
        {
            if (mode >= LMS_CALC_MODE_MAX)
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

        /**
         * @brief 计算右端项均匀范数 ||b||₂（归一化残差的分母）
         * @param mat 系数矩阵（用于取列数，保证与残差口径一致）
         * @param rhs 右端项
         * @return ||b||₂ = sqrt(Σ_{i<ColSize} rhs[i]²)
         * @note 使用前务必确保 mat / rhs 长度足够匹配
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
         * @note 调用前需保证矩阵已装满（rowSize == colSize）且 mat/x/rhs 长度匹配，
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

    protected:
        LinearSolverBase() = default;

        /**
         * @brief 求解前矩阵与向量通用校验（不含各求解器参数范围）
         * @param mat 系数矩阵
         * @param x 解向量
         * @param rhs 右端项
         * @return LMS_OK 全部通过；否则返回第一个失败条件对应的错误码
         * @note 具体求解器在本校验后追加各自的构造参数范围检查
         */
        unsigned short PreSolveCheck(const SparseMatrix<T> &mat, const Vec &x, const Vec &rhs) const
        {
            return Check({
                {[&]() -> bool
                 {
                     return (!mat.IsValid()) || (mat.ColSize() == 0);
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
            });
        }

        /**
         * @brief 迭代过程中的发散检查（各求解器共享）
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

    protected:
        T finalRes_ = T{};         // 最终残差
        size_t iterCnt_ = {0};     // 实际迭代次数
        bool converged_ = {false}; // 是否收敛

        CalcProcessMode mode_ = {LMS_CALC_MODE_SERIAL}; // 计算模式
    };

    // ====================== SOR求解器实现 ======================
    /**
     * @brief SOR（逐次超松弛）求解器实现
     * @tparam T 浮点类型（float/double）
     * @details 提供串行与红黑染色并行两种求解模式，通过 SetCalcProcessMode 切换：
     *          - 串行模式（默认）：经典逐行 Gauss-Seidel/SOR 迭代；
     *          - 并行模式：红黑染色并行 SOR（需 -fopenmp）。
     *            将未知量按矩阵拓扑二分（红/黑），每轮迭代拆成两个并行 pass——
     *            红点只读黑点旧值、黑点读红点新值，隐式 barrier 同步，保留顺序
     *            收敛语义；非二分图（含奇环）自动退化为串行。
     *          - 自动模式：规模足够大且多线程时自动启用并行，否则走串行。
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

        ~SORSolver() = default;

        /**
         * @brief 求解线性方程组 A x = b
         * @param mat 系数矩阵（数据容器，求解器只读）
         * @param x 输入输出参数：进入时为零初值或上次解（作初值猜测），
         *          退出时为解向量（长度必须 >= mat.ColSize()）
         * @param rhs 右端项
         * @return LMS_OK 收敛；
         *         LMS_INI_ERR 矩阵未初始化；
         *         LMS_VEC_LEN_ERR x/rhs 长度不足；
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
            case LMS_CALC_MODE_SERIAL:
            {
                result = SerialSolve(mat, x, rhs);
                break;
            }
            case LMS_CALC_MODE_MULTI_PROCESS_MODE:
            {
                result = MultiProcSolve(mat, x, rhs);
                break;
            }
            case LMS_CALC_MODE_AUTO:
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

        /// @brief 查询求解器类型
        LightMatrixSolverType GetSolverType() const override
        {
            return LMS_SOLVER_TYPE_SOR;
        }

        /**
         * @brief 重新设置 SOR 求解参数（构造后亦可调整）
         * @param maxIter 最大迭代次数，合法范围 [MIN_ITER, MAX_ITER]
         * @param omega 松弛因子（1.0 即 Gauss-Seidel），合法范围 (0, MAX_ALPHAW]
         * @param tol 收敛容差（基于归一化残差），合法范围 (0, MAX_SET_TOL]
         * @note 参数合法性在下一次 Solve 时统一校验
         */
        void SetSorSolverPara(size_t maxIter, T omega, T tol = static_cast<T>(DEFAULT_TOL))
        {
            maxIter_ = maxIter;
            omega_ = omega;
            tol_ = tol;
        }

    private:
        /**
         * @brief 求解前参数与状态校验
         * @param mat 系数矩阵
         * @param x 解向量
         * @param rhs 右端项
         * @return LMS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short SolverParaCheck(const SparseMatrix<T> &mat, const Vec &x, const Vec &rhs) const
        {
            auto result = this->PreSolveCheck(mat, x, rhs);
            if (result != LMS_OK)
            {
                return result;
            }
            // 追加 SOR 特有参数（含松弛因子 omega）范围检查
            return Check({
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
         * @brief 串行 SOR 迭代
         * @param mat 系数矩阵
         * @param x 解向量（原地更新）
         * @param rhs 右端项
         * @return 收敛（LMS_OK）/ 发散（LMS_DIVERGED）/ 未收敛（LMS_NOT_CONVERGED）
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
            T bNorm = this->ComputeUniformResidual(mat, rhs);
            T res0 = this->ComputeResidual(mat, x, rhs);

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
                    T res = this->ComputeResidual(mat, x, rhs);
                    this->finalRes_ = res;
                    this->iterCnt_ = it + 1;
                    auto err = this->DivergenceCheck(res, res0, eps);
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
         * @brief 红黑染色并行 SOR 迭代
         * @param mat 系数矩阵
         * @param x 解向量（原地更新）
         * @param rhs 右端项
         * @return 收敛（LMS_OK）/ 发散（LMS_DIVERGED）/ 未收敛（LMS_NOT_CONVERGED）
         * @note 矩阵非二分图（含奇环）时自动退化为串行求解
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
            T bNorm = this->ComputeUniformResidual(mat, rhs);
            T res0 = this->ComputeResidual(mat, x, rhs);

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
                    T res = this->ComputeResidual(mat, x, rhs);
                    this->finalRes_ = res;
                    this->iterCnt_ = it + 1;
                    auto err = this->DivergenceCheck(res, res0, eps);
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
         * @brief 构建红黑染色（BFS 二染色），并缓存结果
         * @param mat 系数矩阵（读取 CSR 拓扑）
         * @return true=染色成功（二分图）；false=含奇环，非二分图（调用方退化为串行）
         * @note 基于 StructureRevision 缓存：矩阵结构未变化时直接复用已染色结果
         */
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

    // ====================== Jacobi求解器实现 ======================
    /**
     * @brief Jacobi（雅可比）求解器实现
     * @tparam T 浮点类型（float/double）
     * @details 每轮用上一轮旧值同步更新全部未知量，天然无数据依赖：
     *          - 串行模式（默认）：逐未知量计算新值并暂存，整轮结束统一回写；
     *          - 并行模式：两阶段 parallel for（计算新值 / 统一回写），无同步依赖；
     *          - 自动模式：规模足够大且多线程时自动并行，否则串行。
     *          收敛轮数通常多于 SOR/GS，但单轮并行度高、无同步依赖。
     */
    template <typename T>
    class JacobiSolver : public LinearSolverBase<T>
    {
        using Arr = T *;
        using Vec = std::vector<T>;
        using Mat = std::vector<std::vector<T>>;

    public:
        /**
         * @brief 构造 Jacobi 求解器
         * @param maxIter 最大迭代次数，合法范围 [MIN_ITER, MAX_ITER]
         * @param tol 收敛容差（基于归一化残差），合法范围 (0, MAX_SET_TOL]
         * @note 默认串行模式；并行 / 自动模式通过 SetCalcProcessMode 切换
         */
        JacobiSolver(size_t maxIter, T tol = static_cast<T>(DEFAULT_TOL))
            : maxIter_(maxIter), tol_(tol)
        {
        }

        ~JacobiSolver() = default;

        /**
         * @brief 求解线性方程组 A x = b
         * @param mat 系数矩阵（数据容器，求解器只读）
         * @param x 输入输出参数：进入时为零初值或上次解（作初值猜测），
         *          退出时为解向量（长度必须 >= mat.ColSize()）
         * @param rhs 右端项
         * @return LMS_OK 收敛；
         *         LMS_INI_ERR 矩阵未初始化；
         *         LMS_VEC_LEN_ERR x/rhs 长度不足；
         *         LMS_SPARSE_ROW_NOT_FULL 矩阵未填充满至方阵；
         *         LMS_SOLVE_SET_ERR 构造参数 iter/tol 非法；
         *         LMS_NOT_CONVERGED 迭代跑满未达标（x 为最后迭代值，不可信）；
         *         LMS_DIVERGED 发散（NaN/Inf 或残差爆炸）；
         *         其余为参数/状态校验错误码
         * @note 每 10 次迭代检测一次残差；收敛或发散时提前返回，x 保留当时结果；
         *       计算模式由 SetCalcProcessMode 决定（串行 / 并行 / 自动）
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
            case LMS_CALC_MODE_SERIAL:
            {
                result = SerialSolve(mat, x, rhs);
                break;
            }
            case LMS_CALC_MODE_MULTI_PROCESS_MODE:
            {
                result = MultiProcSolve(mat, x, rhs);
                break;
            }
            case LMS_CALC_MODE_AUTO:
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

        /// @brief 查询求解器类型
        LightMatrixSolverType GetSolverType() const
        {
            return LMS_SOLVER_TYPE_JACOBI;
        }

        /**
         * @brief 重新设置 Jacobi 求解参数（构造后亦可调整）
         * @param maxIter 最大迭代次数，合法范围 [MIN_ITER, MAX_ITER]
         * @param tol 收敛容差（基于归一化残差），合法范围 (0, MAX_SET_TOL]
         * @note 参数合法性在下一次 Solve 时统一校验
         */
        void SetJacobiSolverPara(size_t maxIter, T tol = static_cast<T>(DEFAULT_TOL))
        {
            maxIter_ = maxIter;
            tol_ = tol;
        }

    private:
        /**
         * @brief 求解前参数与状态校验
         * @param mat 系数矩阵
         * @param x 解向量
         * @param rhs 右端项
         * @return LMS_OK 全部条件通过；否则返回第一个失败条件对应的错误码
         */
        unsigned short SolverParaCheck(const SparseMatrix<T> &mat, const Vec &x, const Vec &rhs) const
        {
            auto result = this->PreSolveCheck(mat, x, rhs);
            if (result != LMS_OK)
            {
                return result;
            }
            // 追加 Jacobi 参数范围检查
            return Check({
                {[&]() -> bool
                 {
                     return ((this->maxIter_ < MIN_ITER) ||
                             (this->maxIter_ > MAX_ITER) ||
                             (this->tol_ <= 0) || (this->tol_ > static_cast<T>(MAX_SET_TOL)));
                 },
                 LMS_SOLVE_SET_ERR},
            });
        }

        /**
         * @brief 串行 Jacobi 迭代
         * @param mat 系数矩阵
         * @param x 解向量（原地更新）
         * @param rhs 右端项
         * @return 收敛（LMS_OK）/ 发散（LMS_DIVERGED）/ 未收敛（LMS_NOT_CONVERGED）
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
            T bNorm = this->ComputeUniformResidual(mat, rhs);
            T res0 = this->ComputeResidual(mat, x, rhs);

            for (size_t it = 0; it < maxIter_; it++)
            {
                Vec newX(n); // 新解暂存（避免每轮拷贝 rhs）
                for (size_t i = 0; i < n; i++)
                {
                    size_t start = rowPtr[i];
                    size_t end = rowPtr[i + 1];
                    T sum = rhs[i]; // 局部标量累加
                    for (size_t j = start; j < end; j++)
                    {
                        if (colIdx[j] != i)
                        {
                            sum -= value[j] * x[colIdx[j]];
                        }
                    }
                    newX[i] = sum;
                }

                for (size_t i = 0; i < n; i++)
                {
                    x[i] = newX[i] / value[diagIdx[i]];
                }

                if ((it + 1) % checkEvery == 0)
                {
                    T res = this->ComputeResidual(mat, x, rhs);
                    this->finalRes_ = res;
                    this->iterCnt_ = it + 1;
                    auto err = this->DivergenceCheck(res, res0, eps);
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
         * @brief 并行 Jacobi 迭代
         * @param mat 系数矩阵
         * @param x 解向量（原地更新）
         * @param rhs 右端项
         * @return 收敛（LMS_OK）/ 发散（LMS_DIVERGED）/ 未收敛（LMS_NOT_CONVERGED）
         * @note 单线程时退化为串行求解
         */
        unsigned short MultiProcSolve(const SparseMatrix<T> &mat, Vec &x, const Vec &rhs)
        {
            if (maxThread_ == 1)
            {
                return SerialSolve(mat, x, rhs); // 单线程退化串行
            }

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
            T bNorm = this->ComputeUniformResidual(mat, rhs);
            T res0 = this->ComputeResidual(mat, x, rhs);

            for (size_t it = 0; it < maxIter_; it++)
            {
                Vec tmpRhs(n); // 新解暂存（避免每轮拷贝 rhs）
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < n; i++)
                {
                    size_t start = rowPtr[i];
                    size_t end = rowPtr[i + 1];
                    T sum = rhs[i]; // 局部标量累加
                    for (size_t j = start; j < end; j++)
                    {
                        if (colIdx[j] != i)
                        {
                            sum -= value[j] * x[colIdx[j]];
                        }
                    }
                    tmpRhs[i] = sum;
                }
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif
                for (size_t i = 0; i < n; i++)
                {
                    x[i] = tmpRhs[i] / value[diagIdx[i]];
                }

                if ((it + 1) % checkEvery == 0)
                {
                    T res = this->ComputeResidual(mat, x, rhs);
                    this->finalRes_ = res;
                    this->iterCnt_ = it + 1;
                    auto err = this->DivergenceCheck(res, res0, eps);
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

    private:
        size_t maxIter_; // 最大迭代次数
        T tol_;          // 收敛容差

        // 并行Jacobi相关参数
        unsigned short maxThread_ = {1}; // 最大线程数
    };
}

#endif // LIGHT_MATRIX_SOLVER_H
