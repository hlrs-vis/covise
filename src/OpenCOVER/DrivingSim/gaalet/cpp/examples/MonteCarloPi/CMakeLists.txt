cmake_minimum_required (VERSION 2.6.3)
project (MonteCarloPi)

include_directories ("../../../include/cpp")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

add_executable (MonteCarloPi MonteCarloPi.cpp)
