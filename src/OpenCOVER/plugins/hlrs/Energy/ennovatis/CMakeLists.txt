option(DEBUG_ENNOVATIS "Debug Ennovatis" OFF)

configure_file("build_options.h.in" "include/build_options.h")

set(SOURCES
	building.cpp
	csv.cpp
	date.cpp
	json.cpp
	rest.cpp
	sax.cpp
)

set(HEADER
	building.h
	csv.h
	date.h
	json.h
	rest.h
	sax.h
	channel.h
)

set(Name "Ennovatis")
add_library(${Name} ${SOURCES} ${HEADER})
add_dependencies(${Name} coCurlHTTPClient coThreadUtil)
target_link_libraries(${Name} 
	PRIVATE nlohmann_json::nlohmann_json
	PRIVATE coCurlHTTPClient
	PRIVATE coThreadUtil
	PRIVATE coReadCSVUtil)

# tell compiler to generate position independent code => -fPIC
set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Add generated header path to include paths.
target_include_directories(${Name} PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/include")