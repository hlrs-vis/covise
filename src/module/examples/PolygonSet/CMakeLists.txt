SET(HEADERS
  PolygonSet.h
)
SET(SOURCES
  PolygonSet.cpp
)
covise_add_module(Examples PolygonSet ${EXTRASOURCES} ${SOURCES} ${HEADERS})
