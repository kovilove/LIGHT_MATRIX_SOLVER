/**
 * @file demo6.cpp
 * @brief 示例代码
 * @note  测试按行动态添加矩阵接口效果
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include "../LMS/lms.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using std::cout;
using std::endl;

constexpr size_t n = 10;

int main()
{
    using namespace LMS;
    // mat - 构造稀疏矩阵（数据容器）
    SparseMatrix<float> mat;

    mat.Reset(n);
    for (int i = n - 1; i >= 0; i--)
    {
        std::vector<float> vec(n, {});
        vec[i] = float(i + 1);
        if (i < n - 1)
        {
            vec[i + 1] += float(i + 1);
        }
        if (i > 0)
        {
            vec[i - 1] += float(i + 1);
        };

        mat.AddRow(vec, i);
    }
    std::vector<float> vec(n, -1.0);
    mat.AddRow(vec, 4);
    // mat.AddRow(vec, 0);
    /*
    1 1 0 0 0 0 0 0 0 0
    2 2 2 0 0 0 0 0 0 0
    0 0 3 3 3 0 0 0 0 0
    0 0 0 4 4 4 0 0 0 0
    0 0 0 0 5 5 5 0 0 0
    0 0 0 0 0 6 6 6 0 0
    0 0 0 0 0 0 7 7 7 0
    0 0 0 0 0 0 0 8 8 8
    0 0 0 0 0 0 0 0 9 9
    */
    std::cout << mat.Finalize() << std::endl;
    auto val = mat.Values();
    auto idx = mat.ColIdx();
    auto row = mat.RowPtr();
    std::cout << "rowsize: " << row.size() << std::endl;
    for (int i = 0; i < idx.size(); i++)
    {
        std::vector<float> vec(n, {});
        for (size_t j = row[i]; j < row[i + 1]; j++)
        {
            vec[idx[j]] = val[j];
        }
        for (auto v : vec)
        {
            std::cout << " " << std::setw(2) << v;
        }
        std::cout << std::endl;
    }

    return 0;
}
