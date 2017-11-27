using(OpenGL)
covise_find_package(OSMesa)
if (NOT OSMESA_FOUND)
    return()
endif()

include_directories(${OSMESA_INCLUDE_DIR})

set(HEADERS
    WindowTypeMesa.h
)

set(SOURCES
    WindowTypeMesa.cpp
)

cover_add_plugin(WindowTypeMesa ${HEADERS} ${SOURCES} ${OSMESA_LIBRARY})
