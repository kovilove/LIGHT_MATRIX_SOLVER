/**
 * @file example2.cpp
 * @brief 示例代码
 * @note  1000 × 1000 稀疏矩阵求解
 */

#include <iostream>
#include <thread>
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
    // mtx - 构造稀疏矩阵
    SparseMatrix<float> mtx;
    size_t row = 0;

    // x - 待求解量
    std::vector<float> x1(num, {});
    std::vector<float> x2(num, {});

    // rhs - 右端项
    std::vector<float> rhs(num, 10);
    rhs[0] = 20;       // 设置第一个元素为 20，增加非均匀性
    rhs[num - 1] = 20; // 设置最后一个元素为 20，增加非均匀性

    // SetRhs - 添加右端项
    mtx.SetRhs(rhs);

    // 设置矩阵尺寸
    mtx.Reset(num);

    // AddMtx - 逐行添加向量
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
            mtx.AddRow(row_vec);
        }
    }

    // 设置为串行模式
    mtx.SetMultiProcessMode(0);
    // SOR_Solve -  SOR求解
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = mtx.SOR_Solve(x1, 100, 0.5);
    if (result != LSS_OK)
    {
        cout << "calculate err, errcode: " << result << endl;
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "SOR_Solve Serial time: " << duration.count() << " microseconds" << endl;

    // 设置为并行模式
    mtx.SetMultiProcessMode(1);
    // SOR_Solve -  SOR求解
    auto t2 = std::chrono::high_resolution_clock::now();
    auto result2 = mtx.SOR_Solve(x2, 100, 0.5);
    if (result2 != LSS_OK)
    {
        cout << "calculate err, errcode: " << result2 << endl;
        return 1;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    std::cout << "SOR_Solve Multi-Process time: " << duration2.count() << " microseconds" << endl;

    // 打印机求解量
    // for (auto &ele : x1)
    // {
    //     cout << ele << "\n";
    // }
    return 0;
}