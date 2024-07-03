find_package(nlohmann_json)

if (NOT nlohmann_json_FOUND)
    message("nlohmann_json not found")
endif()

if (NOT CURL_FOUND)
    message("CURL not found")
endif()

file(GLOB_RECURSE SOURCES "*.cpp" "*.h" "core/*.cpp" "core/*.h")
add_library(coGraphQL OBJECT ${SOURCES})
add_dependencies(coGraphQL coCurlHTTPClient)
target_link_libraries(
    coGraphQL 
    PRIVATE coCurlHTTPClient
    PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(coGraphQL PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)