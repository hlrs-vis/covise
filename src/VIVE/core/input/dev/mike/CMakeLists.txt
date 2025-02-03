INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(HEADERS
   MikeDriver.h
   ../legacy/serialcom.h
)

SET(SOURCES
   MikeDriver.cpp
   ../legacy/serialcom.cpp
)

vive_add_plugin(vvInput_mike)
