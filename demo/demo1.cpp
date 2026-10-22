/**
 * @file demo1.cpp
 * @brief 示例代码
 * @note  10 × 10 稀疏矩阵求解（SORSolver）
 */

#include <iostream>
#include <thread>
#include <chrono>
#include "../LMS/lms.h"

using std::cout;
using std::endl;

constexpr size_t num = 10;

int main()
{
    using namespace LMS;
    // mat - 构造稀疏矩阵
    SparseMatrix<float> mat;
    size_t row = 0;
    // x - 待求解量
    std::vector<float> x(num, {});
    // rhs - 右端项
    std::vector<float> rhs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 设置矩阵尺寸
    mat.Reset(num);
    // AddRow - 按行号逐行动态添加（COO 装配，行序任意）
    for (int i = num - 1; i >= 0; i--)
    {
        std::vector<float> vec(num, {});
        vec[i] = float(i + 1) * 2;
        if (i < num - 1)
        {
            vec[i + 1] += float(i + 1);
        }
        if (i > 0)
        {
            vec[i - 1] += float(i + 1);
        };

        mat.AddRow(vec, i);
    }

    std::cout << mat.Finalize() << std::endl;

    // SORSolver - SOR 求解（解耦架构：矩阵只管数据，求解器只管算法）
    SORSolver<float> solver(1000, 0.5f, 1e-6f);
    auto t0 = std::chrono::high_resolution_clock::now();

    auto result = solver.Solve(mat, x, rhs);
    if (result != LMS_OK)
    {
        cout << "calculate err";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "SORSolver Solve time: " << duration.count() << " microseconds" << endl;
    std::cout << "residual: " << solver.GetResidual()
              << ", iter: " << solver.GetIterCount() << endl;
    // 打印机求解量
    for (auto &ele : x)
    {
        cout << ele << "\n";
    }

    return 0;
}