option(DEBUG_ENNOVATIS "Debug Ennovatis" OFF)

configure_file("build_options.h.in" "include/build_options.h")

file(GLOB_RECURSE SOURCES "*.cpp" "*.h")
set(Name "Ennovatis")
add_library(${Name} ${SOURCES})
add_dependencies(${Name} coCurlHTTPClient coThreadUtil)
target_link_libraries(${Name} 
	PRIVATE nlohmann_json::nlohmann_json
	PRIVATE coCurlHTTPClient
	PRIVATE coThreadUtil)

# tell compiler to generate position independent code => -fPIC
set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Add generated header path to include paths.
target_include_directories(${Name} PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/include")