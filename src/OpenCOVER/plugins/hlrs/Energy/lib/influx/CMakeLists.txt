set(SOURCES
	rest.cpp
	# apache.cpp
)

set(HEADERS
	rest.h
	# apache.h
)

set(Name "Influx")
add_library(${Name} OBJECT ${SOURCES} ${HEADERS})
target_compile_options(${Name} PRIVATE -fPIC)
add_dependencies(${Name} coCurlHTTPClient coThreadUtil)
target_link_libraries(${Name} 
	coCurlHTTPClient
	coThreadUtil)