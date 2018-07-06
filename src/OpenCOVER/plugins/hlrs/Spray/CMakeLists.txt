covise_find_package(embree 3.0.0)
if (NOT EMBREE_FOUND)
    return()
endif()

include_directories(${EMBREE_INCLUDE_DIRS})

SET(HEADERS
)
SET(SOURCES
gen.cpp
nozzle.cpp
nozzlemanager.cpp
Spray.cpp
nodevisitorvertex.cpp
)

cover_add_plugin(Spray ${HEADERS} ${SOURCES})

set_target_properties(Spray PROPERTIES OUTPUT_NAME Spray)
target_link_libraries(Spray ${EMBREE_LIBRARIES})

