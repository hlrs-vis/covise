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

set(HEADERS
	building.h
	csv.h
	date.h
	json.h
	rest.h
	sax.h
	channel.h
)

set(Name "Ennovatis")
add_library(${Name} STATIC ${SOURCES} ${HEADERS})
target_compile_options(${Name} PRIVATE -fPIC)
add_dependencies(${Name} coCurlHTTPClient coThreadUtil)
target_link_libraries(
	${Name} 
	PRIVATE nlohmann_json::nlohmann_json
	PRIVATE coCurlHTTPClient
	PRIVATE coThreadUtil
	PRIVATE coReadCSVUtil)

# Add generated header path to include paths.
target_include_directories(${Name} PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/include")

COVISE_INSTALL_TARGET(${Name})