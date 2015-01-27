SET(HEADERS
  LabView.h
)
SET(SOURCES
  LabView.cpp
)
cover_add_plugin(LabView)
TARGET_LINK_LIBRARIES(LabView coVRML)
#COVER_INSTALL_PLUGIN(LabView)
