INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(HEADERS
   windowsHID.h
   vvRawDevice.h
)

SET(SOURCES
   windowsHID.cpp
   vvRawDevice.cpp
)

vive_add_plugin(vvInput_windowsHID)
