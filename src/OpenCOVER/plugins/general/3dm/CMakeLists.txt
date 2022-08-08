#OpenCOVER plugin to load 3dm files

USING(OPENNURBS)

SET(HEADERS
  3dmPlugin.h
)

SET(SOURCES
  3dmPlugin.cpp
)


cover_add_plugin(3dm ${HEADERS})
COVISE_WNOERROR(3dm)
