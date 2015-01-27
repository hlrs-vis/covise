SET(HEADERS
  IsoSurfacePlugin.h
  IsoSurfaceInteraction.h
  IsoSurfacePoint.h
)
SET(SOURCES
  IsoSurfacePlugin.cpp
  IsoSurfaceInteraction.cpp
  IsoSurfacePoint.cpp
)
cover_add_plugin(IsoSurfacePlugin)
SET_TARGET_PROPERTIES(IsoSurfacePlugin PROPERTIES OUTPUT_NAME IsoSurface)
