USING(OSGEARTH)

SET(HEADERS
  EarthPlugin.h
  EarthLayers.h
  EarthViewpoint.h
  EarthLayerManager.h
  EarthViewpointManager.h
)
SET(SOURCES
  EarthPlugin.cpp
  EarthLayers.cpp
  EarthViewpoint.cpp
  EarthLayerManager.cpp
  EarthViewpointManager.cpp
)


cover_add_plugin(Earth)
