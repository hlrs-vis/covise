if (NOT COVISE_USE_OSMESA)
    return()
endif()

covise_find_package(OSMesa REQUIRED)
using(OpenGL)

include_directories(${OSMESA_INCLUDE_DIR})

set(HEADERS
    WindowTypeMesa.h
)

set(SOURCES
    WindowTypeMesa.cpp
)

cover_add_plugin(WindowTypeMesa ${HEADERS} ${SOURCES} ${OSMESA_LIBRARY})
