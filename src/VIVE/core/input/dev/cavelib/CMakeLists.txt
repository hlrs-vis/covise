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

cover_add_plugin(input_cavelib)
