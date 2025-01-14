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

cover_add_plugin(input_mousebuttons)
