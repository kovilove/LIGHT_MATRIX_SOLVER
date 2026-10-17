/**
 * @file example2.cpp
 * @brief 示例代码
 * @note  100 × 100 稀疏矩阵求解
 */

#include <iostream>
#include "../LSS/light_sor_solver.hpp"

using std::cout;
using std::endl;

constexpr size_t num = 100;

int main()
{
    using namespace LSS;
    // mtx - 构造稀疏矩阵
    SparseMatrix<float> mtx;
    size_t row = 0;
    // x - 待求解量
    std::vector<float> x(num, {});
    // rhs - 右端项
    std::vector<float> rhs(num, 10);

    // 设置矩阵尺寸
    mtx.Reset(num);
    // AddMtx - 逐行添加向量
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
        mtx.AddRow(row_vec);
    }
    // SetRhs - 添加右端项
    mtx.SetRhs(rhs);
    // SOR_Solve -  SOR求解
    auto result = mtx.SOR_Solve(x, 100, 0.5);
    if (result != LSS_OK)
    {
        cout << "calculate err, errcode: " << result << endl;
        return 1;
    }
    // 打印机求解量
    for (auto &ele : x)
    {
        cout << ele << "\n";
    }
    return 0;
}