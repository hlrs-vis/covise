#OpenCOVER plugins general TangiblePosition

using(VRML)

SET(HEADERS
  TrackObjects.h
)

SET(SOURCES
  TrackObjects.cpp
)

cover_add_plugin(TrackObjects Vrml97Cover)
