#OpenCOVER plugin to load GLTF and GLB files
USING(TinyGLTF)
USING(DRACO)

SET(HEADERS
  GLTFPlugin.h
  GLTFReader.h
)

SET(SOURCES
  GLTFPlugin.cpp
)

cover_add_plugin(GLTF ${HEADERS})
COVISE_WNOERROR(GLTF)
