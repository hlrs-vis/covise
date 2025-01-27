INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(HEADERS
   MouseButtons.h
   MouseButtonsDriver.h
)

SET(SOURCES
   MouseButtons.cpp
   MouseButtonsDriver.cpp
)

vive_add_plugin(vvInput_mousebuttons)
