option(TEST_ENERGYCAMPUS "Test EnergyCampus" OFF)

if(NOT TEST_ENERGYCAMPUS)
    return()
endif()

cmake_minimum_required(VERSION 3.14)
project(EnergyCampusTest)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(GTest REQUIRED)

enable_testing()

add_executable(Test test_energy.cpp)
target_link_libraries(Test EnergyUtils Ennovatis GTest::gtest_main)
add_definitions(-DENERGYCAMPUS_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

include(GoogleTest)
gtest_discover_tests(Test)