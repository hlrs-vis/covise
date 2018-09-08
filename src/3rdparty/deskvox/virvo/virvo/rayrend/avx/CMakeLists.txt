check_include_file("immintrin.h" AVX_SUPPORTED)

if(NOT AVX_SUPPORTED)
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
    add_definitions(-D__AVX__)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()

add_definitions(-DHAVE_CONFIG_H)
add_definitions(-DVV_ARCH_AVX=1)

deskvox_add_library(rayrendavx
    ${RAYREND_HEADERS}
    ${RAYREND_SOURCES}
)
