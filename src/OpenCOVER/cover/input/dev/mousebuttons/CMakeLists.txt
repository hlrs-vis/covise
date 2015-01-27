INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
  "${COVISEDIR}/src/kernel/OpenVRUI"
  "${COVISEDIR}/src/renderer/OpenCOVER"
)

SET(HEADERS
   MouseButtons.h
   MouseButtonsDriver.h
)

SET(SOURCES
   MouseButtons.cpp
   MouseButtonsDriver.cpp
)

cover_add_plugin(input_mousebuttons)
