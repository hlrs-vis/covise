USING(OpenSceneGraph)
set(SOURCES
    color.cpp
    osgUtils.cpp
    math.cpp
)

set(HEADERS
    color.h
    osgUtils.h
    math.h
)

set(Name "utils")
add_library(${Name} STATIC ${SOURCES} ${HEADERS})
target_link_libraries(${Name} ${EXTRA_LIBS})
target_compile_options(${Name} PRIVATE -fPIC)
target_include_directories(
    ${Name} 
    PRIVATE ${OPENSCENEGRAPH_INCLUDE_DIRS}
	PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include"
    )
add_library(core::utils ALIAS ${Name})
COVISE_INSTALL_TARGET(${Name})