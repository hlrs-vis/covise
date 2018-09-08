deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

set(RAYREND_HEADERS
    ../../vvraycaster.h
)

set(RAYREND_SOURCES
    ../../vvraycaster.cpp
)

add_definitions(-DHAVE_CONFIG_H)

deskvox_add_library(rayrendfpu
    ${RAYREND_HEADERS}
    ${RAYREND_SOURCES}
)

