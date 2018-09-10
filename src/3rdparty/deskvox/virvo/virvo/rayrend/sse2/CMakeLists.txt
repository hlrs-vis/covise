check_include_file("emmintrin.h" SSE2_SUPPORTED)

if(NOT SSE2_SUPPORTED)
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
add_definitions(-DHAVE_CONFIG_H)
add_definitions(-DVV_ARCH_SSE2=1)

deskvox_add_library(rayrendsse2
    ${RAYREND_HEADERS}
    ${RAYREND_SOURCES}
)
