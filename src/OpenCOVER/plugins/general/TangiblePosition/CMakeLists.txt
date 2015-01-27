#OpenCOVER plugins general TangiblePosition

SET(HEADERS
  TangiblePositionPlugin.h
)

SET(SOURCES
  TangiblePositionPlugin.cpp
)

cover_add_plugin(TangiblePosition Vrml97Cover ${COVISE_APPL_LIBRARY} ${COVISE_SHM_LIBRARY})
