#OpenCOVER plugins general TangiblePosition

using(VRML)

SET(HEADERS
  TangiblePositionPlugin.h
)

SET(SOURCES
  TangiblePositionPlugin.cpp
)

if(COVISE_BUILD_ONLY_COVER_PREREQ)
    cover_add_plugin(TangiblePosition Vrml97Cover)
else()
    cover_add_plugin(TangiblePosition Vrml97Cover ${COVISE_APPL_LIBRARY} ${COVISE_SHM_LIBRARY})
endif()
