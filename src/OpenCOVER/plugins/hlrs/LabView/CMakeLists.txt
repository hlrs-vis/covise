using(VRML)

SET(HEADERS
  LabView.h
)
SET(SOURCES
  LabView.cpp
)
cover_add_plugin(LabView)
TARGET_LINK_LIBRARIES(LabView)
#COVER_INSTALL_PLUGIN(LabView)
