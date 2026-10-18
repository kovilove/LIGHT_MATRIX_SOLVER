/**
 * @file example2.cpp
 * @brief 示例代码
 * @note  100 × 100 稀疏矩阵求解
 */

#include <iostream>
#include "../LMS/light_matrix_solver.hpp"

using std::cout;
using std::endl;

constexpr size_t num = 100;

int main()
{
    using namespace LMS;
    // mat - 构造稀疏矩阵
    SparseMatrix<float> mat;
    // x - 待求解量
    std::vector<float> x(num, {});
    // rhs - 右端项
    std::vector<float> rhs(num, 10);

    // 设置矩阵尺寸
    mat.Reset(num);
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
        mat.AddRow(row_vec);
    }

    // SORSolver - SOR 求解（解耦架构）
    SORSolver<float> solver(100, 0.5f, 1e-6f);
    auto result = solver.Solve(mat, x, rhs);
    if (result != LMS_OK)
    {
        cout << "calculate err, errcode: " << result << endl;
        return 1;
    }
    std::cout << "residual: " << solver.GetResidual()
              << ", iter: " << solver.GetIterCount() << endl;
    // 打印机求解量
    for (auto &ele : x)
    {
        cout << ele << "\n";
    }
    return 0;
}