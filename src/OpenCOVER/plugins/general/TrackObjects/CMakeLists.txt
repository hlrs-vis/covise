#OpenCOVER plugins general TangiblePosition

using(VRML)

SET(HEADERS
  TrackObjects.h
)

SET(SOURCES
  TrackObjects.cpp
)

cover_add_plugin(TrackObjects Vrml97Cover ${COVISE_APPL_LIBRARY} ${COVISE_SHM_LIBRARY})
