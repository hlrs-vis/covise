#OpenCOVER plugins general JTOpen

USING(JT)

SET(HEADERS
  findNodeVisitor.h
  JTOpenPlugin.h
)

SET(SOURCES
  findNodeVisitor.cpp
  JTOpenPlugin.cpp
)

cover_add_plugin(JTOpen)
COVISE_WNOERROR(JTOpen)
