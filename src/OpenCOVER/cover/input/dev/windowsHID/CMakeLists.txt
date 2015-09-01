INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(HEADERS
   windowsHID.h
   coRawDevice.h
)

SET(SOURCES
   windowsHID.cpp
   coRawDevice.cpp
)

cover_add_plugin(input_windowsHID)
