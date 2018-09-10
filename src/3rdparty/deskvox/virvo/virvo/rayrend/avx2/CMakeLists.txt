check_include_file("immintrin.h" AVX2_SUPPORTED)

if(NOT AVX2_SUPPORTED)
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
    add_definitions(-D__AVX2__)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

add_definitions(-DHAVE_CONFIG_H)
add_definitions(-DVV_ARCH_AVX2=1)

deskvox_add_library(rayrendavx2
    ${RAYREND_HEADERS}
    ${RAYREND_SOURCES}
)
