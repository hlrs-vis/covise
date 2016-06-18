# OpenCOVER PluginUtil (coOpenPluginUtil)
USING(BOOST)

USING(VTK optional)
IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()

ADD_DEFINITIONS(-DCOVISE_PLUGIN_UTIL)

INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(PLUGINUTIL_HEADERS
  coColorBar.h
  ColorBar.h
  ModuleFeedbackManager.h
  ModuleFeedbackPlugin.h
  BoxSelection.h
  BoxSelectionInteractor.h
  PluginMessageTypes.h
  coArrow.h
  coPlane.h
  coVR1DTransInteractor.h
  coVR2DTransInteractor.h
  coVR3DRotInteractor.h
  coVR3DTransInteractor.h
  coVR3DTransRotInteractor.h
  FileReference.h
  SimReference.h
  coInstanceRenderer.h
  coBaseCoviseInteractor.h
  coSphere.h
  ProgramCache.h
  coLOD.h
  coSensor.h
  GenericGuiObject.h
  MultiChannelDrawer.h
)

SET(SOURCES
  coColorBar.cpp
  ColorBar.cpp
  ModuleFeedbackManager.cpp
  ModuleInteraction.cpp
  ModuleFeedbackPlugin.cpp
  BoxSelection.cpp
  BoxSelectionInteractor.cpp
  coArrow.cpp
  coPlane.cpp
  coVR1DTransInteractor.cpp
  coVR2DTransInteractor.cpp
  coVR3DRotInteractor.cpp
  coVR3DTransInteractor.cpp
  coVR3DTransRotInteractor.cpp
  FileReference.cpp
  SimReference.cpp
  coInstanceRenderer.cpp
  coBaseCoviseInteractor.cpp
  coSphere.cpp
  ProgramCache.cpp
  coLOD.cpp
  coSensor.cpp
  GenericGuiObject.cpp
  MultiChannelDrawer.cpp
)

ADD_COVISE_LIBRARY(coOpenPluginUtil SHARED )
TARGET_LINK_LIBRARIES(coOpenPluginUtil coOpenCOVER coOpenVRUI coOSGVRUI
   ${COVISE_GRMSG_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES})
COVISE_INSTALL_TARGET(coOpenPluginUtil)
COVISE_INSTALL_HEADERS(PluginUtil ${PLUGINUTIL_HEADERS})
qt_use_modules(coOpenPluginUtil Core)
