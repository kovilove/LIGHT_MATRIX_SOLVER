# LIGHT_MATRIX_SOLVER

一个轻量级、**header-only**、C++17 的稀疏矩阵线性求解头文件库。基于 **CSR（Compressed Sparse Row，压缩稀疏行）** 稀疏存储，采用**数据 / 算法解耦 + 策略模式**架构：矩阵只负责存储与组装，求解算法通过抽象基类 `LinearSolverBase` 可插拔扩展。当前内置 **SOR（逐次超松弛）** 及红黑染色并行实现，计划后续按需扩展 Krylov 子空间法（CG / BiCGSTAB / GMRES）、多重网格（AMG / GAMG）等求解器，上层调用无需改动。适用于 CFD 等大规模稀疏线性系统场景——在简单场景下不想引入完整线性代数库时可尝试使用。

## 特性

- **单头文件、零依赖**：把 `LMS/light_matrix_solver.hpp` 拷进工程即可使用，C++17 编译，无任何外部依赖（并行模式可选 OpenMP）
- **CSR 稀疏存储**：内存高效，适合大规模稀疏系统；对角元位置单独索引，迭代求解 O(1) 访问
- **可插拔求解器架构**：`SparseMatrix`（数据容器）与 `LinearSolverBase`（抽象基类）+ `SORSolver`（实现）解耦，策略模式便于运行时切换求解器，未来扩展 GAMG / PCG / CG 等无需改动上层
- **串行 + 红黑染色并行双模式**：串行 SOR 开箱即用（松弛因子可调，`1.0` 即 Gauss-Seidel）；并行模式将未知量按矩阵拓扑二分（红/黑），每轮迭代拆成两个并行 pass（OpenMP），保留顺序收敛语义；含奇环的非二分图自动退化为串行
- **自动模式（AUTO）**：规模足够大且多线程时自动启用并行，否则走串行，免去手动开关
- **右端项外置**：矩阵只负责存储与组装，右端项 `b` 由调用方通过 `Solve(mat, x, rhs)` 传入，职责更清晰
- **健壮性检查与发散保护**：NaN/Inf 校验、对角元非零检查、参数合法性校验、完整错误码体系；迭代中周期性监测残差，相对 / 绝对双重发散判定，不收敛时明确报错而非返回垃圾解
- **便捷的矩阵组装**：逐行（`AddRow`）、批量追加（`AddMultiRow`）、覆盖式整表（`AddMat`）三种输入方式
- **模板化浮点类型**：支持 `float` / `double` 等，编译期 `static_assert` 强制校验

## 当前限制

- 仅支持方阵；内置求解器为 SOR，无预条件子与直接法后备，松弛因子需手动调整
- 红黑并行要求矩阵拓扑为二分图（含奇环时退化为串行），且需 `-fopenmp` 编译

## 目录结构

```
LIGHT_MATRIX_SOLVER/
├── LMS/
│   └── light_matrix_solver.hpp   # 整个库（header-only）
├── demo/
│   ├── demo1.cpp                 # 10×10 稀疏系统示例
│   ├── demo2.cpp                 # 三对角系统示例
│   └── demo3.cpp                 # 10000×10000 稀疏系统：串行 vs 红黑并行对照与一致性校验
└── README.md
```

## 快速上手

```cpp
#include <iostream>
#include <vector>
#include "LMS/light_matrix_solver.hpp"

using namespace LMS;

int main()
{
    SparseMatrix<double> mat;

    // 1. 设置矩阵尺寸（方阵，> 2）
    mat.Reset(3);

    // 2. 组装矩阵（对角元必须非零）
    size_t errRow = 0;
    mat.AddMat({
        {4.0, 1.0, 0.0},
        {1.0, 3.0, 1.0},
        {0.0, 1.0, 4.0},
    }, errRow);

    // 3. 右端项 b 与解向量 x（右端项由调用方持有）
    std::vector<double> rhs = {1.0, 2.0, 3.0};
    std::vector<double> x(3, 0.0);

    // 4. SOR 求解 Ax = b（w = 1.2，最多 100 次迭代）
    SORSolver<double> solver(100, 1.2);             // 构造求解器（数据与算法解耦）
    unsigned short ret = solver.Solve(mat, x, rhs); // 传入矩阵与右端项求解

    if (ret == LMS_OK)
    {
        // x = [0.15, 0.4, 0.65]
        for (double v : x) std::cout << v << "\n";
    }
    return 0;
}
```

任意 C++17 编译器即可编译（并行示例需 `-fopenmp`）：

```bash
g++ -std=c++17 -O2 demo/demo1.cpp -o demo1
g++ -std=c++17 -O2 -fopenmp demo/demo3.cpp -o demo3
```

## API 一览

### 矩阵（`SparseMatrix`，数据容器）

| 函数 | 说明 |
|---|---|
| `Reset(colSize)` | 重置为 `colSize × colSize` 方阵并预留内存 |
| `Clear()` | 清空一切，恢复未初始化状态 |
| `ClearMat()` | 仅清空矩阵系数（保留矩阵尺寸，可重新填充） |
| `AddRow(vec)` | 添加一行（稠密输入，内部压缩为 CSR） |
| `AddMultiRow(mat, errRow)` | 批量追加；失败时 `errRow` 定位出错行 |
| `AddMat(mat, errRow)` | 覆盖式批量组装；带尺寸预检 |
| `IsValid()` / `IsFullMatrix()` | 查询初始化状态 / 是否已装满可求解 |
| `ColSize()` / `RowSize()` / `NotZeroElemNum()` | 查询矩阵属性 |
| `Values()` / `ColIdx()` / `RowPtr()` / `DiagIdx()` / `Eps()` / `StructureRevision()` | CSR 结构只读访问器（供求解器使用） |

### 求解器（`LinearSolverBase` 抽象基类 / `SORSolver` 实现）

| 类 / 成员 | 说明 |
|---|---|
| `SORSolver(maxIter, omega, tol)` | 构造 SOR 求解器 |
| `solver.Solve(mat, x, rhs)` | 求解 `Ax = b`；`x` 输入初值猜测、输出解，`mat` 为数据容器，`rhs` 为右端项 |
| `solver.GetSolverType()` | 查询求解器类型（`LightMatrixSolverType`，当前为 `LMS_SOR_SOLVER`） |
| `solver.GetResidual()` / `GetIterCount()` | 查询最终残差 / 实际迭代次数 |
| `solver.IsConverged()` | 查询是否收敛 |
| `solver.SetCalcProcessMode(mode)` / `GetCalcProcessMode()` | 设置 / 查询计算模式（串行 / 并行 / 自动） |
| `solver.SetSorSolverPara(maxIter, omega, tol)` | 设置最大迭代次数 / 松弛因子 / 收敛容差 |

## 求解器类型

| 枚举 | 说明 |
|---|---|
| `LMS_DEFAULT` | 默认（未指定具体类型） |
| `LMS_SOR_SOLVER` | SOR（逐次超松弛）求解器 |

## 扩展指南：新增求解器

本库以「数据 / 算法解耦 + 策略模式」设计，新增一种求解算法无需改动矩阵与上层调用：

1. 继承 `LinearSolverBase<T>`；
2. 实现纯虚接口 `Solve(mat, x, rhs)` 与 `GetSolverType()`（在 `LightMatrixSolverType` 中注册新类型）；
3. 复用基类默认提供的 `GetResidual()` / `GetIterCount()` / `IsConverged()` / `SetCalcProcessMode()` 等公共接口与状态；
4. 通过 `SparseMatrix` 的只读访问器（`Values/ColIdx/RowPtr/DiagIdx`）访问CSR矩阵数据。

```cpp
template <typename T>
class CGSolver : public LinearSolverBase<T> {
public:
    unsigned short Solve(const SparseMatrix<T>& mat, Vec& x, const Vec& rhs) override;
    LightMatrixSolverType GetSolverType() const override { return LMS_CG_SOLVER; }
};
```

上层只需持有 `LinearSolverBase<T>` 指针，运行时切换不同求解器。

## SORSolver 参数

| 参数 | 含义 | 合法范围 |
|---|---|---|
| `maxIter` | 最大迭代次数 | `[10, 1000]` |
| `omega` | 松弛因子（`1.0` 即 Gauss-Seidel） | `(0, 2]`，SPD 矩阵推荐 `1.0 ~ 1.5` |
| `tol` | 收敛容差（归一化残差 `‖Ax-b‖/‖b‖`） | `(0, 0.1]`，默认 `1e-6` |

## 计算模式

| 模式 | 说明 |
|---|---|
| `LMS_SERIAL_MODE` | 串行逐行 Gauss-Seidel/SOR（默认） |
| `LMS_MULTI_PROCESS_MODE` | 红黑染色并行 SOR（需 `-fopenmp`）；非二分图自动退串行 |
| `LMS_AUTO_MODE` | 规模足够大且多线程时自动并行，否则串行 |

## 错误码

所有 API 均返回错误码，完整带注释清单见头文件。
主要错误码：`LMS_OK`（成功）、`LMS_INI_ERR`、`LMS_VEC_LEN_ERR`、`LMS_MAT_SIZE_ERR`、
`LMS_DIAG_ELEM_ZERO`、`LMS_NAN_ELEM_ERR`、`LMS_SPARSE_ROW_FULL`、
`LMS_SPARSE_ROW_NOT_FULL`、`LMS_SOLVE_SET_ERR`、`LMS_NOT_CONVERGED`、`LMS_DIVERGED`、`LMS_MODE_ERR`。

## 收敛性说明

- 严格对角占优是 SOR 收敛的**充分条件而非必要条件**；求解器会预检并仅作提示，不阻断
- 对对称正定（SPD）矩阵，`omega ∈ (0, 2)` 内任意取值均保证收敛
- 每 10 次迭代监测一次残差；收敛或发散时提前返回，`x` 保留当时的迭代结果
- 红黑并行保留与串行一致的收敛语义（`demo3` 提供串行 / 并行结果一致性校验）

## 编译要求

- C++17 及以上
- 无任何外部依赖（并行模式需支持 OpenMP 的编译器，如 GCC `-fopenmp`）

## 许可证

[MIT](LICENSE)

## 贡献

个人学习代码，欢迎提出问题和提交 Issue 或者 Pull Request。
