option(TEST_ENERGY "Test Energy" OFF)

if(NOT TEST_ENERGY)
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

set(Name "EnergyTest")
add_executable(
    ${Name} ${TEST_SOURCES} 
)

target_include_directories(${Name} 
    PRIVATE ${CMAKE_SOURCE_DIR}/src/OpenCOVER/plugins/hlrs/Energy 
    PRIVATE ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

target_link_libraries(
    ${Name}
    PRIVATE coCurlHTTPClient 
    PRIVATE coThreadUtil 
    PRIVATE coStringUtil 
    PRIVATE coReadCSVUtil 
    PRIVATE core::simulation 
    PRIVATE nlohmann_json::nlohmann_json 
    PRIVATE GTest::gtest_main 
    ${EXTRA_LIBS}
)

include(GoogleTest)
gtest_discover_tests(${Name})
