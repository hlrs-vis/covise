#OpenCOVER plugins general VideoLOD

SET(HEADERS
  SLOD.h
  VideoLOD.h
  VideoLODPlugin.h
)

SET(SOURCES
  SLOD.cpp
  VideoLOD.cpp
  VideoLODPlugin.cpp
)

cover_add_plugin(VideoLOD)
