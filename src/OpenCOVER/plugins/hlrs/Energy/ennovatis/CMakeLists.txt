find_package(nlohmann_json 3.11.2)

if (NOT nlohmann_json_FOUND)
    message("nlohmann_json not found")
    return()
endif()

option(DEBUG_ENNOVATIS "Debug Ennovatis" OFF)

configure_file("build_options.h.in" "include/build_options.h")

file(GLOB_RECURSE SOURCES "*.cpp" "*.h")
add_library(Ennovatis OBJECT $<TARGET_OBJECTS:EnergyUtils> ${SOURCES})
target_link_libraries(Ennovatis PRIVATE nlohmann_json::nlohmann_json)

# tell compiler to generate position independent code => -fPIC
set_target_properties(Ennovatis PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Add generated header path to include paths.
target_include_directories(Ennovatis PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/include")