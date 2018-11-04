#OpenCOVER plugins general Tracer

SET(HEADERS
  TracerFreePoints.h
  TracerInteraction.h
  TracerLine.h
  TracerPlane.h
  TracerPlugin.h
)

SET(SOURCES
  TracerFreePoints.cpp
  TracerInteraction.cpp
  TracerLine.cpp
  TracerPlane.cpp
  TracerPlugin.cpp
)

if (COVISE_BUILD_ONLY_COVER_PREREQ)
cover_add_plugin(TracerPlugin coOpenPluginUtil)
else()
include_directories(../../covise)
cover_add_plugin(TracerPlugin coOpenPluginUtil CovisePluginUtil ${COVISE_ALG_LIBRARY})
endif()
#TARGET_LINK_LIBRARIES()
SET_TARGET_PROPERTIES(TracerPlugin PROPERTIES OUTPUT_NAME Tracer)
