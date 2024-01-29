option(TEST_ENERGYCAMPUS "Test energy" OFF)

if(NOT TEST_ENERGYCAMPUS)
    return()
endif()

cmake_minimum_required(VERSION 3.14)
project(EnergyCampusTest)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG        main
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(SOURCES ../REST.cpp)
set(HEADERS ../REST.h)

add_executable(Test test_energy.cpp)
target_sources(Test PRIVATE ${SOURCES})
target_include_directories(Test PRIVATE ../${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(Test GTest::gtest_main ${CURL_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(Test)