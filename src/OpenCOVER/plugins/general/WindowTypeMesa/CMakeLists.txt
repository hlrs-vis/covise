using(OpenGL)
covise_find_package(OSMesa)
if (NOT OSMESA_FOUND)
    return()
endif()

set(HEADERS
    WindowTypeMesa.h
)

set(SOURCES
    WindowTypeMesa.cpp
)

cover_add_plugin(WindowTypeMesa ${HEADERS} ${SOURCES} ${QRC_SOURCES} ${UI_SOURCES})
