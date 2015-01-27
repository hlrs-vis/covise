cmake_minimum_required (VERSION 2.6.3)
project (tests)

include_directories ("../../include/cpp0x")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif (NOT CMAKE_BUILD_TYPE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

file (GLOB test_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
foreach (test_source ${test_sources})
  if(${test_source} STREQUAL "Ginac.cpp")
   message("Excluding Ginac.cpp")
  else(${test_source} STREQUAL "Ginac.cpp")
   message("Adding ${test_source}")
   string (REPLACE ".cpp" "" test_prefix ${test_source})
   add_executable (${test_prefix} ${test_source})
  endif(${test_source} STREQUAL "Ginac.cpp")
endforeach (test_source)

message("Adding Ginac.cpp")
add_executable(Ginac Ginac.cpp)
target_link_libraries(Ginac ginac cln dl)


