CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(configureVsCodeSettings)
# Hint: Set CMAKE_PREFIX_PATH or nlohmann_json_DIR to covise/src/3rdparty/nlohmann_json if not found automatically
# set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_SOURCE_DIR}/../../src/3rdparty/nlohmann_json")
# find_package(nlohmann_json REQUIRED)
add_subdirectory(
    ${CMAKE_SOURCE_DIR}/../../src/3rdparty/nlohmann_json
    ${CMAKE_BINARY_DIR}/nlohmann_json-build
)
add_executable(configureVsCodeSettings configureVsCodeSettings.cpp)
target_link_libraries(configureVsCodeSettings PRIVATE nlohmann_json)