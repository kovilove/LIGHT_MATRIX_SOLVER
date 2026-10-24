# =============================================================================
# LIGHT_MATRIX_SOLVER - 稀疏矩阵线性求解头文件库（header-only）
#
# 定义 INTERFACE 库目标 LMS（别名 LMS::LMS）。header-only 库无需编译产物，
# 仅向外暴露：头文件目录、C++17 要求、可选 OpenMP 并行求解支持。
#
# 本文件自包含、可整体移植：将整个 LMS 目录 copy 到外部工程的任意子目录，
# 在外部 CMakeLists 中 add_subdirectory(<copy 后的路径>/LMS) 即可复用，
# 不依赖 demo、仓库根目录或其他文件。
#
# 使用方式一：子目录嵌入（推荐，供 CFD_SIMPLE 等工程复用）
#   # 把 LMS 目录放到你的工程下，然后：
#   add_subdirectory(<你的工程路径>/LMS)     # 或 add_subdirectory(LMS) 若放在根目录
#   target_link_libraries(your_target PRIVATE LMS)
#   之后即可 #include "lms.h"
#
# 使用方式二：安装 + find_package
#   cmake -S LMS -B build -DCMAKE_INSTALL_PREFIX=<prefix> && cmake --install build
#   find_package(LMS REQUIRED)
#   target_link_libraries(your_target PRIVATE LMS::LMS)
#
# 开关：LMS_USE_OPENMP（默认 ON）——并行求解模式（SOR 红黑染色 / Jacobi 直接并行）需要 OpenMP
# =============================================================================

cmake_minimum_required(VERSION 3.15)
project(LMS VERSION 1.0.0 LANGUAGES CXX)

if(TARGET LMS)
    return()
endif()

add_library(LMS INTERFACE)
add_library(LMS::LMS ALIAS LMS)

# header-only：头文件位于本目录，将整个目录 copy 到外部工程后 add_subdirectory(LMS) 即可使用
target_include_directories(LMS INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>   # 构建期：本目录（可整体移植）
    $<INSTALL_INTERFACE:include>                     # 安装期：include 目录
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
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
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
