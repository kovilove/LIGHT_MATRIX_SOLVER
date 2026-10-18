/**
 * @file demo3.cpp
 * @brief 示例代码
 * @note  10000 × 10000 稀疏矩阵求解，演示解耦架构（SparseMatrix 数据 + SORSolver 算法）
 *        矩阵组装使用 OpenMP 并行加速；求解当前为串行 SOR（红黑并行将在后续版本加入）
 */

#include <iostream>
#include <chrono>
#include "../LSS/light_sor_solver.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using std::cout;
using std::endl;

constexpr size_t num = 50000;

int main()
{
    using namespace LSS;
    // mat - 构造稀疏矩阵（数据容器）
    SparseMatrix<float> mat;

    // x - 待求解量
    std::vector<float> x1(num, {});
    std::vector<float> x2(num, {});

    // rhs - 右端项
    std::vector<float> rhs(num, 10);
    rhs[0] = 20;       // 增加非均匀性
    rhs[num - 1] = 20; // 增加非均匀性

    // 设置矩阵尺寸（必须先 Reset，再 SetRhs）
    mat.Reset(num);

    // 逐行组装矩阵（OpenMP 并行加速填充，不影响求解算法）
#ifdef _OPENMP
#pragma omp parallel for ordered schedule(static)
#endif
    for (size_t i = 0; i < num; ++i)
    {
        std::vector<float> row_vec(num, 0);
        row_vec[i] = 5;
        if (i < num - 1)
        {
            row_vec[i + 1] = 1;
            row_vec[i] += row_vec[i + 1];
        }
        if (i > 0)
        {
            row_vec[i - 1] = 1;
            row_vec[i] += row_vec[i - 1];
        }
#pragma omp ordered
        {
            mat.AddRow(row_vec);
        }
    }

    // SORSolver
    SORSolver<float> solver(100, 0.5f, 1e-6f);

    // 当前串行 SOR
    solver.SetCalcProcessMode(SOR_SERIAL_MODE);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = solver.Solve(mat, x1, rhs);
    if (result != LSS_OK)
    {
        cout << "solve err, errcode: " << result << endl;
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // 当前并行 SOR
    solver.SetCalcProcessMode(SOR_MULTI_PROCESS_MODE);
    auto t2 = std::chrono::high_resolution_clock::now();
#ifdef _OPENMP
    result = solver.Solve(mat, x2, rhs);
    if (result != LSS_OK)
    {
        cout << "solve err, errcode: " << result << endl;
        return 1;
    }
#endif
    auto t3 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "Serial SOR time: " << duration.count() << " ms"
              << ", residual: " << solver.GetResidual()
              << ", iter: " << solver.GetIterCount() << endl;

#ifdef _OPENMP
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    std::cout << "Multiproc SOR time: " << duration.count() << " ms"
              << ", residual: " << solver.GetResidual()
              << ", iter: " << solver.GetIterCount() << endl;
#endif

// 校验结果
#ifdef _OPENMP
#pragma omp prarllel for
    for (size_t i = 0; i < num; i++)
    {
        if (x1[i] - x2[i] > 0.0001)
        {
#pragma omp critical
            {
                std::cout << "i: " << i << " x1:" << x1[i] << " x2:" << x2[i] << std::endl;
            }
        }
    }
#endif
    return 0;
}
