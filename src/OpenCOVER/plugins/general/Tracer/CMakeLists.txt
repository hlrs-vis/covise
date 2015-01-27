#OpenCOVER plugins general Tracer
USE_VTK(OPTIONAL)

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

cover_add_plugin(TracerPlugin coOpenPluginUtil COVISEPluginUtil ${COVISE_ALG_LIBRARY})
#TARGET_LINK_LIBRARIES()
SET_TARGET_PROPERTIES(TracerPlugin PROPERTIES OUTPUT_NAME Tracer)
