include_directories( /home/georg/embree/embree_libs/include/embree3/)

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
target_link_libraries(Spray /home/georg/embree/embree_libs/lib/libembree3.so.3)

