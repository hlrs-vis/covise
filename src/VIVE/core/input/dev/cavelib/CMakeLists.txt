# Plugin for caveLib 2.6


INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(HEADERS
   CAVELIBDriver.h
)

SET(SOURCES
   CAVELIBDriver.cpp
)

vive_add_plugin(vvInput_cavelib)
