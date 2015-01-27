cmake_minimum_required (VERSION 2.6.3)
project (MonteCarloPi)

include_directories ("../../../include/cpp0x")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

add_executable (MonteCarloPi MonteCarloPi.cpp)
