<<<<<<< HEAD
USING(OpenGL)
USING(PNG)
covise_find_package(OSMesa)
if (NOT OSMESA_FOUND)
=======
if (NOT COVISE_USE_OSMESA)
>>>>>>> c7be9f77e08addb0cae52532f4f14015d9934d6c
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
