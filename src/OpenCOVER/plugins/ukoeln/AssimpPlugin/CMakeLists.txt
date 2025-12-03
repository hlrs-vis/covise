# OpenCOVER plugin to load 3D model files via Assimp library
USING(Assimp)

SET(HEADERS
  AssimpPlugin.h
  AssimpReader.h
)

SET(SOURCES
  AssimpPlugin.cpp
)

cover_add_plugin(AssimpPlugin ${HEADERS} ${SOURCES})
COVISE_WNOERROR(AssimpPlugin)
