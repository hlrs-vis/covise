
SET(HEADERS
  MapLinkPlugin.h
)
SET(SOURCES
  MapLinkPlugin.cpp
)
cover_add_plugin(MapLink)
target_link_libraries(MapLink coVRGeoData)
