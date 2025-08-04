option(TEST_ENERGYCAMPUS "Test EnergyCampus" OFF)

if(NOT TEST_ENERGYCAMPUS)
    return()
endif()

cmake_minimum_required(VERSION 3.14)
project(EnergyTest)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG "v1.17.0"
    GIT_SHALLOW TRUE #copy version and not git history

)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SOURCES "src/*.cpp")

enable_testing()

add_executable(
    Test ${TEST_SOURCES} 
    $<TARGET_OBJECTS:Ennovatis>
    $<TARGET_OBJECTS:Apache_Arrow>)
target_include_directories(Test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/OpenCOVER/plugins/hlrs/Energy
)
target_link_libraries(Test Arrow::arrow_shared coCurlHTTPClient nlohmann_json::nlohmann_json coThreadUtil coStringUtil coReadCSVUtil GTest::gtest_main)
add_definitions(-DENERGYCAMPUS_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

include(GoogleTest)
gtest_discover_tests(Test)