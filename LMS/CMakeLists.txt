# =============================================================================
# LIGHT_MATRIX_SOLVER - 稀疏矩阵线性求解头文件库（header-only）
#
# 定义 INTERFACE 库目标 LMS（别名 LMS::LMS）。header-only 库无需编译产物，
# 仅向外暴露：头文件目录、C++17 要求、可选 OpenMP 并行求解支持。
#
# 使用方式一：子目录嵌入（推荐，供 CFD_SIMPLE 等工程复用）
#   add_subdirectory(<path>/LIGHT_MATRIX_SOLVER)   # 顶层脚本，含 demo 构建选项
#   # 或 add_subdirectory(<path>/LMS)              # 仅库本身
#   target_link_libraries(your_target PRIVATE LMS)
#   之后即可 #include "LMS/lms.h"
#
# 使用方式二：安装 + find_package
#   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<prefix> && cmake --install build
#   find_package(LMS REQUIRED)
#   target_link_libraries(your_target PRIVATE LMS::LMS)
#
# 开关：LMS_USE_OPENMP（默认 ON）——并行求解模式（SOR 红黑染色 / Jacobi 直接并行）需要 OpenMP
# =============================================================================

cmake_minimum_required(VERSION 3.15)

if(TARGET LMS)
    return()
endif()

add_library(LMS INTERFACE)
add_library(LMS::LMS ALIAS LMS)

# header-only：仅声明头文件目录与 C++ 标准
target_include_directories(LMS INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>   
    $<INSTALL_INTERFACE:include>                        
)
target_compile_features(LMS INTERFACE cxx_std_17)

# 可选 OpenMP：作为 INTERFACE 使用要求，随 target_link_libraries 传播给调用方
option(LMS_USE_OPENMP "Enable OpenMP for parallel solve modes" ON)
if(LMS_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(LMS INTERFACE OpenMP::OpenMP_CXX)
        # GCC/MinGW 下 OpenMP::OpenMP_CXX 仅携带编译期 -fopenmp，链接期需显式补上，
        # 否则调用方链接时会出现 omp_* / GOMP_parallel 未定义引用
        if(OpenMP_CXX_FLAGS)
            target_link_options(LMS INTERFACE ${OpenMP_CXX_FLAGS})
        endif()
    endif()
endif()

# ---- 安装 / 导出（供 find_package 消费） ----
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# 记录 OpenMP 依赖标记，注入包配置文件：消费端 find_package(LMS) 时自动 find_dependency(OpenMP)
# （导出的 LMS_EXPORT_TARGETS.cmake 引用了 OpenMP::OpenMP_CXX，消费端必须先定义该目标）
if(OpenMP_CXX_FOUND)
    set(LMS_OPENMP_DEP "include(CMakeFindDependencyMacro)\nfind_dependency(OpenMP)")
else()
    set(LMS_OPENMP_DEP "")
endif()

install(TARGETS LMS EXPORT LMS_EXPORT_TARGETS)
install(FILES
    lms.h
    lms_sparse_matrix.hpp
    lms_linear_solver.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LMS
)

# 生成包配置文件，使 find_package(LMS) 可用
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/LMSConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/LMSConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LMSConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LMS
)

install(EXPORT LMS_EXPORT_TARGETS
    FILE LMS_EXPORT_TARGETS.cmake
    NAMESPACE LMS::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LMS
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/LMSConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/LMSConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LMS
)
