#OpenCOVER plugins general Sky

USING(OSGEPHEMERIS)
USING(VRML)

SET(HEADERS
  Sky.h
)

SET(SOURCES
  Sky.cpp
)

cover_add_plugin(Sky)
