INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
  "${COVISEDIR}/src/kernel/OpenVRUI"
  "${COVISEDIR}/src/renderer/OpenCOVER"
)

SET(HEADERS
   MikeDriver.h
   ../legacy/serialcom.h
)

SET(SOURCES
   MikeDriver.cpp
   ../legacy/serialcom.cpp
)

cover_add_plugin(input_mike)
