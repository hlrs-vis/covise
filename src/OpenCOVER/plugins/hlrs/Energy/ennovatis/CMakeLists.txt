USING(CURL)

option(DEBUG_ENNOVATIS "Debug Ennovatis" OFF)

configure_file("build_options.h.in" "include/build_options.h")

file(GLOB_RECURSE SOURCES "*.cpp" "*.h")
add_library(Ennovatis OBJECT ${SOURCES})

target_link_libraries(Ennovatis PRIVATE ${CURL_LIBRARIES})
# tell compiler to generate position independent code => -fPIC
set_target_properties(Ennovatis PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Add generated header path to include paths.
target_include_directories(Ennovatis PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/include")