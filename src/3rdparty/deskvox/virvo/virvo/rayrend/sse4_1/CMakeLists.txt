check_include_file("smmintrin.h" SSE4_1_SUPPORTED)

if(NOT SSE4_1_SUPPORTED)
    return()
endif()

deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

set(RAYREND_HEADERS
    ../../vvraycaster.h
)

set(RAYREND_SOURCES
    ../../vvraycaster.cpp
)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    add_definitions(-D__SSE4_1__)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()

add_definitions(-DHAVE_CONFIG_H)
add_definitions(-DVV_ARCH_SSE4_1=1)

deskvox_add_library(rayrendsse4_1
    ${RAYREND_HEADERS}
    ${RAYREND_SOURCES}
)
