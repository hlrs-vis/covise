#OpenCOVER plugins general Cal3DNode

USING(OSGCAL)
USING(VRML)

SET(HEADERS
  Cal3DPlugin.h
)

SET(SOURCES
  Cal3DPlugin.cpp
)

cover_add_plugin(Cal3DNode Vrml97Cover)
covise_wnoerror(Cal3DNode)
