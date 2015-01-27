cmake_minimum_required (VERSION 2.6.3)
project (benchmarks)

include_directories ("../../include/cpp")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

file (GLOB benchmark_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
foreach (benchmark_source ${benchmark_sources})
  string (REPLACE ".cpp" "" benchmark_prefix ${benchmark_source})
  add_executable (${benchmark_prefix} ${benchmark_source})
endforeach (benchmark_source)
