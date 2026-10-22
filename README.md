# LIGHT_MATRIX_SOLVER

一个轻量级、**header-only**、C++17 的稀疏矩阵线性求解头文件库，由聚合头 `LMS/lms.h` 统一入口，包含 `LMS/lms_sparse_matrix.hpp`（稀疏矩阵存储与装配）+ `LMS/lms_linear_solver.hpp`（迭代求解器）。基于 **CSR（Compressed Sparse Row，压缩稀疏行）** 稀疏存储，采用**数据 / 算法解耦 + 策略模式**架构：矩阵只负责存储与组装，求解算法通过抽象基类 `LinearSolverBase` 可插拔扩展。矩阵以 **COO 坐标三元组动态装配**（`AddRow(vec,row)` / `AddValue` / `Finalize`）统一压缩为标准 CSR；当前内置 **SOR（逐次超松弛）** 与 **Jacobi（雅可比）** 两种迭代求解器（均含 OpenMP 并行实现），计划后续按需扩展 Krylov 子空间法（CG / BiCGSTAB / GMRES）、多重网格（AMG / GAMG）等求解器，上层调用无需改动。适用于 CFD 等大规模稀疏线性系统场景——在简单场景下不想引入完整线性代数库时可尝试使用。

## 特性

- **纯头文件、零依赖**：`lms.h`（聚合入口）+ `lms_sparse_matrix.hpp`（存储 / 装配）+ `lms_linear_solver.hpp`（求解器）拷进工程即可使用，C++17 编译，无任何外部依赖（并行模式可选 OpenMP）
- **CSR 稀疏存储**：内存高效，适合大规模稀疏系统；对角元位置单独索引，迭代求解 O(1) 访问
- **COO 动态装配**：以 `AddRow(vec, row)` / `AddValue` 按任意行序添加坐标三元组（同一 `(row,col)` 可重复累加），`Finalize` 统一排序、压缩并转换为标准 CSR；适合 CFD 中先拼内部节点、后拼边界的非顺序组装
- **可插拔求解器架构**：`SparseMatrix`（数据容器）与 `LinearSolverBase`（抽象基类）+ 多求解器实现（`SORSolver` / `JacobiSolver`）解耦，策略模式便于运行时切换，未来扩展 GAMG / PCG / CG 等无需改动上层
- **SOR 串行 + 红黑染色并行**：串行 SOR 开箱即用（松弛因子可调，`1.0` 即 Gauss-Seidel）；并行模式将未知量按矩阵拓扑二分（红/黑），每轮迭代拆成两个并行 pass（OpenMP），保留顺序收敛语义；含奇环的非二分图自动退化为串行
- **Jacobi 串行 + 直接并行**：每轮用上一轮旧值同步更新全部未知量，天然无数据依赖，可直接 `parallel for` 并行；收敛轮数多于 SOR/GS，但单轮并行度高、无同步依赖
- **自动模式（AUTO）**：规模足够大且多线程时自动启用并行，否则走串行，免去手动开关
- **右端项外置**：矩阵只负责存储与组装，右端项 `b` 由调用方通过 `Solve(mat, x, rhs)` 传入，职责更清晰
- **健壮性检查与发散保护**：NaN/Inf 校验、对角元非零检查、参数合法性校验、完整错误码体系；矩阵 / 向量通用校验（`PreSolveCheck`）与发散判定（`DivergenceCheck`）上提基类、全求解器共享；迭代中周期性监测残差，相对 / 绝对双重发散判定，不收敛时明确报错而非返回垃圾解
- **模板化浮点类型**：支持 `float` / `double` 等，编译期 `static_assert` 强制校验

## 当前限制

- 仅支持方阵；内置求解器为 SOR 与 Jacobi，无预条件子与直接法后备；SOR 松弛因子需手动调整，Jacobi 收敛较慢
- 红黑并行要求矩阵拓扑为二分图（含奇环时退化为串行），且需 `-fopenmp` 编译

## 目录结构

```
LIGHT_MATRIX_SOLVER/
├── CMakeLists.txt                # 顶层构建脚本（构建 demo / 供外部 add_subdirectory 嵌入）
├── LMS/
│   ├── CMakeLists.txt            # INTERFACE 库定义（LMS / LMS::LMS，含安装与 find_package 导出）
│   ├── lms.h                    # 聚合头：统一包含下面两个头文件
│   ├── lms_sparse_matrix.hpp    # 稀疏矩阵存储 / 装配（COO 动态装配 + CSR 结构，header-only）
│   ├── lms_linear_solver.hpp    # 求解器：LinearSolverBase + SOR / Jacobi（header-only）
│   └── cmake/LMSConfig.cmake.in # find_package 包配置文件模板
├── demo/
│   ├── demo1.cpp                # 10×10 稀疏系统求解示例（SORSolver）
│   ├── demo2.cpp                # 100×100 稀疏系统求解示例（SORSolver）
│   ├── demo3.cpp                # 30000×30000：串行 vs 红黑并行 SOR + 结果一致性校验
│   ├── demo4.cpp                # 10×10：Jacobi 求解 + 计时示例
│   ├── demo5.cpp                # 10000×10000：串行 vs 并行 Jacobi + 结果一致性校验
│   ├── demo6.cpp                # COO 按行动态装配接口效果示例（重复行累加合并）
│   └── demo7.cpp                # （空文件，待填充新示例）
└── README.md
```

## 快速上手

```cpp
#include <iostream>
#include <vector>
#include "LMS/lms.h"   // 或直接包含 "LMS/lms_linear_solver.hpp"

using namespace LMS;

int main()
{
    SparseMatrix<double> mat;

    // 1. 设置矩阵尺寸（方阵，> 2）
    mat.Reset(3);

    // 2. 以 COO 方式按行动态装配（行序任意；同一 (row,col) 可重复累加）
    mat.AddRow({4.0, 1.0, 0.0}, 0);
    mat.AddRow({1.0, 3.0, 1.0}, 1);
    mat.AddRow({0.0, 1.0, 4.0}, 2);

    // 3. 排序、压缩为标准 CSR（每行对角元必须非零，否则返回 LMS_DIAG_ELEM_ZERO）
    mat.Finalize();

    // 4. 右端项 b 与解向量 x（右端项由调用方持有）
    std::vector<double> rhs = {1.0, 2.0, 3.0};
    std::vector<double> x(3, 0.0);

    // 5. SOR 求解 Ax = b（w = 1.2，最多 100 次迭代）
    SORSolver<double> solver(100, 1.2);             // 构造求解器（数据与算法解耦）
    unsigned short ret = solver.Solve(mat, x, rhs); // 传入矩阵与右端项求解

    if (ret == LMS_OK)
    {
        // 本例精确解为 x = [0.15, 0.4, 0.65]
        for (double v : x) std::cout << v << "\n";
    }
    return 0;
}
```

COO 装配支持任意行序与重复累加，适合 CFD 中先拼内部节点、后拼边界的场景：

```cpp
#include "LMS/lms.h"
using namespace LMS;

SparseMatrix<double> mat;
mat.Reset(3);

mat.AddRow({4.0, 1.0, 0.0}, 0);   // 先填第 0 行
mat.AddRow({2.0, 0.0, 5.0}, 2);   // 再填第 2 行（行序可任意）
mat.AddRow({0.0, 3.0, 0.0}, 1);   // 后填第 1 行
mat.AddRow({4.0, 1.0, 0.0}, 0);   // 重复 (0,0)，Finalize 时自动累加合并

mat.Finalize();                   // 排序压缩 → 标准 CSR，之后即可求解
```

任意 C++17 编译器即可编译（并行示例需 `-fopenmp`）：

```bash
g++ -std=c++17 -O2 demo/demo1.cpp -o demo1
g++ -std=c++17 -O2 -fopenmp demo/demo3.cpp -o demo3
```

## CMake 集成（INTERFACE 接口库）

本库以 header-only 形式对外暴露一个 **INTERFACE 库目标 `LMS`**（别名 `LMS::LMS`），无编译产物，仅传播头文件目录、C++17 要求与可选 OpenMP 支持。

**方式一：子目录嵌入（推荐，供 CFD_SIMPLE 等工程复用）**

```cmake
add_subdirectory(<path>/LIGHT_MATRIX_SOLVER)   # 顶层脚本，含 LMS_BUILD_DEMOS 开关
target_link_libraries(your_target PRIVATE LMS) # 或 PRIVATE LMS::LMS
```

之后即可 `#include "LMS/lms.h"` 使用。OpenMP 作为 INTERFACE 使用要求随链接自动传播给调用方（含链接期 `-fopenmp`，无需调用方手动配置）。

**方式二：安装 + find_package**

```bash
cmake -S . -B build && cmake --install build --prefix <prefix>
```

消费端：

```cmake
find_package(LMS REQUIRED)   # 包配置会自动 find_dependency(OpenMP)
target_link_libraries(your_target PRIVATE LMS::LMS)
```

**独立构建（编译全部 demo 用于验证）：**

```bash
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
```

开关：
| 选项 | 默认 | 说明 |
|---|---|---|
| `LMS_USE_OPENMP` | ON | 并行求解模式（SOR 红黑 / Jacobi 直接并行）需要 OpenMP |
| `LMS_BUILD_DEMOS` | ON | 是否构建 demo 可执行文件（嵌入外部工程时建议关掉） |

## API 一览

### 矩阵（`SparseMatrix`，数据容器）

| 函数 | 说明 |
|---|---|
| `Reset(colSize)` | 重置为 `colSize × colSize` 方阵并预留内存（`colSize > 2`） |
| `Clear()` | 清空一切，恢复未初始化状态 |
| `ClearMat()` | 仅清空矩阵系数（保留矩阵尺寸，可重新填充） |
| `AddRow(vec, row)` | **COO**：动态添加第 `row` 行（行序任意、可重复累加，`Finalize` 时合并） |
| `AddValue(value, col, row)` | **COO**：动态添加单个非零元（可重复累加） |
| `Finalize()` | **COO 收尾**：排序、压缩并转换为标准 CSR，校验对角元非零；之后可求解 |
| `SetValue(value, col, row)` | 覆盖已 Finalize 矩阵指定非零元的值（须已存在） |
| `GetValue(value, col, row)` | 读取指定非零元的值 |
| `ValueIdx(idx, col, row)` | 查询指定非零元在 `value_` 中的下标 |
| `AssemblyStatus()` | 查询装配状态（`INVALID` / `WORKING` / `FINISHED`） |
| `IsValid()` | 查询是否已初始化（Reset 成功且未 Clear） |
| `ColSize()` / `RowSize()` / `NotZeroElemNum()` | 查询矩阵属性 |
| `Values()` / `ColIdx()` / `RowPtr()` / `DiagIdx()` / `Eps()` / `StructureRevision()` | CSR 结构只读访问器（供求解器使用） |

### 求解器（`LinearSolverBase` 抽象基类 / `SORSolver` / `JacobiSolver`）

| 类 / 成员 | 说明 |
|---|---|
| `SORSolver(maxIter, omega, tol)` | 构造 SOR 求解器 |
| `JacobiSolver(maxIter, tol)` | 构造 Jacobi 求解器 |
| `solver.Solve(mat, x, rhs)` | 求解 `Ax = b`；`x` 输入初值猜测、输出解，`mat` 为数据容器，`rhs` 为右端项 |
| `solver.GetSolverType()` | 查询求解器类型（`LightMatrixSolverType`） |
| `solver.GetResidual()` / `GetIterCount()` | 查询最终残差 / 实际迭代次数 |
| `solver.IsConverged()` | 查询是否收敛 |
| `solver.SetCalcProcessMode(mode)` / `GetCalcProcessMode()` | 设置 / 查询计算模式（串行 / 并行 / 自动） |
| `solver.SetSorSolverPara(maxIter, omega, tol)` | 设置最大迭代次数 / 松弛因子 / 收敛容差 |
| `solver.SetJacobiSolverPara(maxIter, tol)` | 设置最大迭代次数 / 收敛容差 |

## 求解器类型

| 枚举 | 说明 |
|---|---|
| `LMS_SOLVER_TYPE_DEFAULT` | 默认（未指定具体类型） |
| `LMS_SOLVER_TYPE_SOR` | SOR（逐次超松弛）求解器 |
| `LMS_SOLVER_TYPE_JACOBI` | Jacobi（雅可比）求解器 |

## 扩展指南：新增求解器

本库以「数据 / 算法解耦 + 策略模式」设计，新增一种求解算法无需改动矩阵与上层调用：

1. 继承 `LinearSolverBase<T>`；
2. 实现纯虚接口 `Solve(mat, x, rhs)` 与 `GetSolverType()`（在 `LightMatrixSolverType` 中追加新枚举，如 `LMS_SOLVER_TYPE_CG`）；
3. 复用基类默认提供的 `GetResidual()` / `GetIterCount()` / `IsConverged()` / `SetCalcProcessMode()` 等公共接口与状态；
4. 复用基类共享的通用校验 `PreSolveCheck` 与发散判定 `DivergenceCheck`，通过 `SparseMatrix` 的只读访问器（`Values/ColIdx/RowPtr/DiagIdx`）访问 CSR 数据。

```cpp
template <typename T>
class CGSolver : public LinearSolverBase<T> {
public:
    unsigned short Solve(const SparseMatrix<T>& mat, Vec& x, const Vec& rhs) override;
    LightMatrixSolverType GetSolverType() const override { return LMS_SOLVER_TYPE_CG; }
};
```

上层只需持有 `LinearSolverBase<T>` 指针，运行时切换不同求解器。

## SORSolver 参数

| 参数 | 含义 | 合法范围 |
|---|---|---|
| `maxIter` | 最大迭代次数 | `[10, 1000]` |
| `omega` | 松弛因子（`1.0` 即 Gauss-Seidel） | `(0, 2]`，SPD 矩阵推荐 `1.0 ~ 1.5` |
| `tol` | 收敛容差（归一化残差 `‖Ax-b‖/‖b‖`） | `(0, 0.1]`，默认 `1e-6` |

## JacobiSolver 参数

| 参数 | 含义 | 合法范围 |
|---|---|---|
| `maxIter` | 最大迭代次数 | `[10, 1000]` |
| `tol` | 收敛容差（归一化残差 `‖Ax-b‖/‖b‖`） | `(0, 0.1]`，默认 `1e-6` |

> 注：Jacobi 每轮同步更新旧值，收敛轮数通常多于 SOR/GS。若在 `MAX_ITER`（默认 1000）上限内仍不收敛，可调大头文件中的全局上限。

## 计算模式

| 模式 | 说明 |
|---|---|
| `LMS_CALC_MODE_SERIAL` | 串行（SOR 逐行 / Jacobi 同步更新，默认） |
| `LMS_CALC_MODE_MULTI_PROCESS_MODE` | 并行（SOR 红黑染色 / Jacobi 直接并行，需 `-fopenmp`）；SOR 非二分图自动退串行 |
| `LMS_CALC_MODE_AUTO` | 规模足够大且多线程时自动并行，否则串行 |

## 错误码

所有 API 均返回错误码，完整带注释清单见头文件 `ErrCode` 枚举。
主要错误码：`LMS_OK`（成功）、`LMS_INI_ERR`、`LMS_VEC_LEN_ERR`、`LMS_OUT_SIZE_LIMIT`、
`LMS_DIAG_ELEM_ZERO`、`LMS_NAN_ELEM_ERR`、`LMS_SPARSE_ROW_NOT_FULL`、
`LMS_STATUS_REJECT`、`LMS_COOTRIPLES_EMPTY`、`LMS_SOLVE_SET_ERR`、
`LMS_NOT_CONVERGED`、`LMS_DIVERGED`、`LMS_MODE_ERR`。

## 收敛性说明

- 严格对角占优是 SOR 收敛的**充分条件而非必要条件**；求解器会预检并仅作提示，不阻断
- 对对称正定（SPD）矩阵，SOR 在 `omega ∈ (0, 2)` 内任意取值均保证收敛
- 严格对角占优或对称正定（SPD）矩阵，Jacobi 迭代保证收敛；收敛速度由谱半径决定，通常慢于 SOR/GS
- 每 10 次迭代监测一次残差；收敛或发散时提前返回，`x` 保留当时的迭代结果
- 红黑并行保留与串行一致的收敛语义（`demo3` 提供串行 / 并行结果一致性校验，实测两路结果一致）

## 性能观察（运行实测）

- `demo3`（30000×30000 SOR）实测串行约 5.9 s、红黑并行约 5.2 s，加速比约 1.1×——三对角系统并行收益有限，主要受单轮浮点负载与同步开销影响
- `demo5`（10000×10000 Jacobi）实测并行耗时高于串行：该规模单轮浮点负载低、同步 / 伪共享开销占比大，并行无明显收益。**大规模（≥10⁵ 未知量）才建议启用并行**
- 上述为当前机器（本机多线程）上的观察记录，非库缺陷；具体加速比随矩阵结构与硬件变化

## 编译要求

- C++17 及以上
- 无任何外部依赖（并行模式需支持 OpenMP 的编译器，如 GCC `-fopenmp`）

## 许可证

[MIT](LICENSE)

## 贡献

个人学习代码，欢迎提出问题和提交 Issue 或者 Pull Request。
