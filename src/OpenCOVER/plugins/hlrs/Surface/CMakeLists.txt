SET(HEADERS
  SurfacePlugin.h
  SurfaceInteraction.h
)
SET(SOURCES
  SurfacePlugin.cpp
  SurfaceInteraction.cpp
)
COVER_ADD_PLUGIN(Surface)
target_link_libraries(Surface CoverMultitouch)
#COVER_INSTALL_PLUGIN(Surface)
