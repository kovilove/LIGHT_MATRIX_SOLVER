# LIGHT_SOR_SOLVER

一个轻量级、**header-only** 的 C++17 稀疏线性方程组求解器，基于 **SOR（逐次超松弛，Successive Over-Relaxation）** 迭代法，采用 **CSR（Compressed Sparse Row，压缩稀疏行）** 存储。适用于 CFD 等稀疏线性系统场景——在一些简单场景下如果不想引入完整的线性代数库，它正好够用。

## 特性

- **单头文件、零依赖**：把 `LSS/light_sor_solver.hpp` 拷进工程即可使用，无任何外部依赖，C++17 编译
- **CSR 稀疏存储**：内存高效，适合大规模稀疏系统；对角元位置单独索引，迭代求解 O(1) 访问
- **可插拔求解器架构**：`SparseMatrix`（数据容器）与 `LinearSolverBase`（抽象基类）+ `SORSolver`（实现）解耦，策略模式便于运行时切换求解器、未来扩展 GAMG/PCG 等
- **串行 SOR 开箱即用**：松弛因子可调，`1.0` 即 Gauss-Seidel；红黑染色并行（OpenMP）规划中，将消除 GS/SOR 迭代依赖、以两 pass 并行更新保留顺序收敛语义
- **健壮性检查与发散保护**：NaN/Inf 校验、对角元非零检查、参数合法性校验、完整错误码体系；迭代中周期性监测残差，相对/绝对双重发散判定，不收敛时明确报错而非返回垃圾解
- **便捷的矩阵组装**：逐行（`AddRow`）、批量追加（`AddMultiRow`）、覆盖式整表（`AddMtx`）三种输入方式；右端项支持向量累加/覆盖与标量累加/覆盖（`AddRhs`/`SetRhs`）
- **模板化浮点类型**：支持 `float`/`double` 等，编译期 `static_assert` 强制校验

## 当前限制

- 仅支持方阵；无预条件子与直接法后备，松弛因子需手动调整
- 当前仅串行求解；红黑染色并行（OpenMP）在规划中

## 目录结构

```
LIGHT_SOR_SOLVER/
├── LSS/
│   └── light_sor_solver.hpp   # 整个库（header-only）
├── example/
│   └── example1.cpp           # 10×10 稀疏系统示例
└── README.md
```

## 快速上手

```cpp
#include <iostream>
#include <vector>
#include "LSS/light_sor_solver.hpp"

using namespace LSS;

int main()
{
    SparseMatrix<double> mat;

    // 1. 设置矩阵尺寸（方阵，> 2）
    mat.Reset(3);

    // 2. 组装矩阵（对角元必须非零）
    size_t errRow = 0;
    mat.AddMtx({
        {4.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 4.0},
    }, errRow);

    // 3. 设置右端项 b
    mat.SetRhs({1.0, 2.0, 3.0});

    // 4. SOR 求解 Ax = b（w = 1.2，最多 100 次迭代）
    std::vector<double> x(3, 0.0);
    SORSolver<double> solver(100, 1.2);              // 构造求解器（数据与算法解耦）
    unsigned short ret = solver.Solve(x, mat);       // 传入矩阵求解

    if (ret == LSS_OK)
    {
        // x = [0.15, 0.4, 0.65]
        for (double v : x) std::cout << v << "\n";
    }
    return 0;
}
```

任意 C++17 编译器即可编译（并行示例需 `-fopenmp`）：

```bash
g++ -std=c++17 -O2 example/demo1.cpp -o demo1
g++ -std=c++17 -O2 -fopenmp example/demo3.cpp -o demo3
```

## API 一览

| 函数 | 说明 |
|---|---|
| `Reset(colSize)` | 重置为 `colSize × colSize` 方阵并预留内存 |
| `Clear()` | 清空一切，恢复未初始化状态 |
| `ClearMtx()` | 仅清空矩阵系数（保留尺寸与右端项，可重新填充） |
| `ClearRhs()` | 右端项清零（保留矩阵结构） |
| `AddRow(vec)` | 添加一行（稠密输入，内部压缩为 CSR） |
| `AddMultiRow(mat, errRow)` | 批量追加；失败时 `errRow` 定位出错行 |
| `AddMtx(mat, errRow)` | 覆盖式批量组装；带尺寸预检 |
| `AddRhs(vec)` / `AddRhs(val)` | 累加右端项：`rhs += vec` / `rhs[i] += val` |
| `SetRhs(vec)` / `SetRhs(val)` | 覆盖右端项：`rhs = vec` / `rhs[i] = val` |
| `IsValid()` / `IsFullMatrix()` | 查询初始化状态 / 是否已装满可求解 |
| `ColSize()` / `RowSize()` / `NotZeroElemNum()` / `Rhs()` | 查询矩阵属性 |

### 求解器（`LinearSolverBase` 抽象基类 / `SORSolver` 实现）

| 类 / 成员 | 说明 |
|---|---|
| `SORSolver(maxIter, omega, tol, mode)` | 构造 SOR 求解器；`mode` 取 `SOR_SERIAL_MODE` / `SOR_MULTI_PROCESS_MODE` / `SOR_AUTO_MODE` |
| `solver.Solve(x, mat)` | 求解 `Ax = b`；`x` 输入初值猜测、输出解，`mat` 为数据容器 |
| `solver.GetResidual()` / `GetIterCount()` | 查询最终残差 / 实际迭代次数 |
| `solver.IsConverged()` | 查询是否收敛 |

未来新增求解器（如 GAMG）只需继承 `LinearSolverBase` 并实现 `Solve`，上层调用无需改动。

### SORSolver 参数

| 参数 | 含义 | 合法范围 |
|---|---|---|
| `iter` | 最大迭代次数 | `[10, 1000]` |
| `alphaW` | 松弛因子（`1.0` 即 Gauss-Seidel） | `(0, 2]`，SPD 矩阵推荐 `1.0 ~ 1.5` |
| `tol` | 收敛容差（归一化残差 `‖Ax-b‖/‖b‖`） | `(0, 0.1]`，默认 `1e-6` |

### 错误码

所有 API 均返回错误码，完整带注释清单见头文件。
主要错误码：`LSS_OK`（成功）、`LSS_INI_ERR`、`LSS_VEC_LEN_ERR`、`LSS_MTX_SIZE_ERR`、
`LSS_DIAG_ELEM_ZERO`、`LSS_NAN_ELEM_ERR`、`LSS_SPARSE_ROW_FULL`、
`LSS_SPARSE_ROW_NOT_FULL`、`LSS_SOLVE_SET_ERR`、`LSS_NOT_CONVERGED`、`LSS_DIVERGED`。

## 收敛性说明

- 严格对角占优是 SOR 收敛的**充分条件而非必要条件**；求解器会预检并仅作提示，不阻断
- 对对称正定（SPD）矩阵，`alphaW ∈ (0, 2)` 内任意取值均保证收敛
- 每 10 次迭代监测一次残差；收敛或发散时提前返回，`x` 保留当时的迭代结果

## 编译要求

- C++17 及以上
- 无任何外部依赖

## 许可证

[MIT](LICENSE)

## 贡献

个人学习代码，欢迎提出问题和提交 Issue 或者 Pull Request。