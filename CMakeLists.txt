# =============================================================================
# LIGHT_MATRIX_SOLVER 顶层构建脚本
#
# 1) 独立构建：cmake -S . -B build && cmake --build build
#    编译全部 demo（demo1~6），用于本地验证
# 2) 嵌入外部工程：add_subdirectory(<path>/LIGHT_MATRIX_SOLVER)
#    得到 INTERFACE 库目标 LMS（别名 LMS::LMS），随后 target_link_libraries(... PRIVATE LMS)
#
# 开关：
#   LMS_USE_OPENMP  并行求解模式需要 OpenMP，默认 ON
#   LMS_BUILD_DEMOS 是否构建 demo 可执行文件，默认 ON（嵌入到外部工程时建议关掉）
# =============================================================================

cmake_minimum_required(VERSION 3.15)
project(LIGHT_MATRIX_SOLVER VERSION 1.0.0 LANGUAGES CXX)

option(LMS_USE_OPENMP "Enable OpenMP for parallel solve modes" ON)
option(LMS_BUILD_DEMOS "Build demo executables" ON)

add_subdirectory(LMS)

if(LMS_BUILD_DEMOS)
    set(LMS_DEMO_SOURCES
        demo/demo1.cpp demo/demo2.cpp demo/demo3.cpp
        demo/demo4.cpp demo/demo5.cpp demo/demo6.cpp
    )
    foreach(src ${LMS_DEMO_SOURCES})
        get_filename_component(name "${src}" NAME_WE)
        add_executable(${name} "${src}")
        target_link_libraries(${name} PRIVATE LMS)
    endforeach()
endif()
