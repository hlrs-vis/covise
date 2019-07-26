
ADD_DEFINITIONS(-DDBGPRINT)

#INCLUDE_DIRECTORIES()

SET(HEADERS
  
)

SET(SOURCES
  GeometryManager.cpp
  ObjectList.cpp
  ObjectManager.cpp
  RenderMain.cpp
  Renderer.cpp
)

SET(EXTRASOURCES
  GeometryManager.h
  ObjectList.h
  ObjectManager.h
  Renderer.h
  RenderManager.h
  Slider_VRML.h
)

ADD_COVISE_MODULE(Renderer VRMLRenderer ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(VRMLRenderer  coAppl coCore coUtil coNet)

COVISE_INSTALL_TARGET(VRMLRenderer)
