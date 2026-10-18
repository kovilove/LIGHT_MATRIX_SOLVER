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

constexpr size_t num = 10000;

int main()
{
    using namespace LSS;
    // mat - 构造稀疏矩阵（数据容器）
    SparseMatrix<float> mat;

    // x - 待求解量
    std::vector<float> x(num, {});

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

    // SORSolver - 解耦后求解（当前串行 SOR）
    SORSolver<float> solver(100, 0.5f, 1e-6f, SOR_SERIAL_MODE);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = solver.Solve(mat, x, rhs);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (result != LSS_OK)
    {
        cout << "solve err, errcode: " << result << endl;
        return 1;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "Serial SOR time: " << duration.count() << " us"
              << ", residual: " << solver.GetResidual()
              << ", iter: " << solver.GetIterCount() << endl;
    return 0;
}
