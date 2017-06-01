using(Virvo)
using(Virvo:osg)
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})

SET(HEADERS
  NeuroPlugin.h
)
SET(SOURCES
  NeuroPlugin.cpp
)


cover_add_plugin(NeuroPlugin ${HEADERS} ${SOURCES} ${CUDA_SOURCES})
set_target_properties(NeuroPlugin PROPERTIES OUTPUT_NAME Neuro)
target_link_libraries(NeuroPlugin ${OPENSCENEGRAPH_LIBRARIES})
