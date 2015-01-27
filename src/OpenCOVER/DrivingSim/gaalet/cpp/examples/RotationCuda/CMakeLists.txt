cmake_minimum_required (VERSION 2.6.3)
project (RotationCuda)

include_directories ("../../../include/cpp")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

find_package(CUDA REQUIRED)
include_directories ("${CUDA_TOOLKIT_INCLUDE}/thrust")
cuda_add_executable (RotationCuda RotationCuda.cu)
