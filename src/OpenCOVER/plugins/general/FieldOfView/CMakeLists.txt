#OpenCOVER plugins general Cal3DNode

USING(VRML)

SET(HEADERS
  FieldOfView.h
)

SET(SOURCES
  FieldOfView.cpp
)

cover_add_plugin(FieldOfView Vrml97Cover)
covise_wnoerror(FieldOfView)
