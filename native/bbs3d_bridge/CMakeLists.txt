cmake_minimum_required(VERSION 3.18)

project(oceansim_bbs3d_bridge LANGUAGES CXX)

set(BBS3D_SOURCE_DIR "" CACHE PATH "Path to the pinned 3D-BBS source tree")
set(BBS3D_BUILD_DIR "" CACHE PATH "Path containing libgpu_bbs3d.so")

if(NOT EXISTS "${BBS3D_SOURCE_DIR}/bbs3d/include/gpu_bbs3d/bbs3d.cuh")
  message(FATAL_ERROR "BBS3D_SOURCE_DIR does not contain the 3D-BBS headers")
endif()

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_library(
  BBS3D_GPU_LIBRARY
  NAMES gpu_bbs3d
  PATHS "${BBS3D_BUILD_DIR}"
  NO_DEFAULT_PATH
  REQUIRED
)

add_library(oceansim_bbs3d SHARED oceansim_bbs3d_bridge.cpp)
target_compile_features(oceansim_bbs3d PRIVATE cxx_std_17)
target_include_directories(
  oceansim_bbs3d
  PRIVATE
    "${BBS3D_SOURCE_DIR}/bbs3d/include"
)
target_link_libraries(
  oceansim_bbs3d
  PRIVATE
    "${BBS3D_GPU_LIBRARY}"
    Boost::filesystem
    Boost::system
    Eigen3::Eigen
    CUDA::cudart
)

# The bridge is intentionally a build-tree dependency.  The Python loader
# preloads the pinned upstream library and this RPATH also makes direct use
# from the build tree work without a system-wide installation.
set_target_properties(
  oceansim_bbs3d
  PROPERTIES
    BUILD_RPATH "${BBS3D_BUILD_DIR}"
    INSTALL_RPATH "$ORIGIN"
)
