# Plugin for Oculus Rift DK2 support

USING(OVR)

INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(HEADERS
   RIFTDriver.h
)

SET(SOURCES
   RIFTDriver.cpp
)

cover_add_plugin(input_rift)
