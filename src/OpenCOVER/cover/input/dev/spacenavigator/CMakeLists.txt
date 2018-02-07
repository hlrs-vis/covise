set(HEADERS
    SpaceNavigatorDriver.h
)

set(SOURCES
    SpaceNavigatorDriver.cpp
)

covise_find_package(HIDAPI)
if (HIDAPI_FOUND)
    add_definitions(-DUSE_HIDAPI)
    include_directories(${HIDAPI_INCLUDE_DIRS})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${HIDAPI_LIBRARIES})
endif()

cover_add_plugin(input_spacenavigator)
