using(VRML)
using(PhysX)

SET(HEADERS
  VrmlNodeCrawler.h
  CrawlerPlugin.h
)
SET(SOURCES
  VrmlNodeCrawler.cpp
  CrawlerPlugin.cpp
)
cover_add_plugin(Crawler)
