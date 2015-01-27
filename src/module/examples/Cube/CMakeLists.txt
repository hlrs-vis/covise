SET(HEADERS
  Cube.h
)
SET(SOURCES
  Cube.cpp
)
covise_add_module(Examples Cube ${EXTRASOURCES} ${SOURCES} ${HEADERS})
