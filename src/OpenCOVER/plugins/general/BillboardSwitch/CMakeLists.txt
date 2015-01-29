# OpenCOVER plugins general AKToolbar

# INCLUDE_DIRECTORIES( )

using(VRML)

SET(HEADERS
  BillboardSwitch.h
)

SET(SOURCES
  BillboardSwitch.cpp
)

cover_add_plugin(BillboardSwitch Vrml97Cover)

