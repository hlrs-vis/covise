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

vive_add_plugin(vvInput_rift)
