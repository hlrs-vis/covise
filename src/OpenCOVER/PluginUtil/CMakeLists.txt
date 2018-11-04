# OpenCOVER PluginUtil (coOpenPluginUtil)
USING(BOOST)
USING(CG optional)
USING(GLEW)

IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()

PROJECT(OpenCOVER CXX C)
ADD_DEFINITIONS(-DCOVISE_PLUGIN_UTIL)

INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

if(COVISE_USE_CUDA)
   covise_find_package(CUDA)

   if(CUDA_FOUND)
      include_directories(${CUDA_INCLUDE_DIRS})
      set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
   endif()
endif()

SET(PLUGINUTIL_HEADERS
  coColorBar.h
  ColorBar.h
  FeedbackManager.h
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
  coVR3DRotCenterInteractor.h
  FileReference.h
  SimReference.h
  coInstanceRenderer.h
  coSphere.h
  ProgramCache.h
  coLOD.h
  coSensor.h
  GenericGuiObject.h
  MultiChannelDrawer.h
  StaticSequence.h
  CudaGraphicsResource.h
  CudaTextureRectangle.h
  Tipsify.h
)

SET(SOURCES
  coColorBar.cpp
  ColorBar.cpp
  FeedbackManager.cpp
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
  coVR3DRotCenterInteractor.cpp
  FileReference.cpp
  SimReference.cpp
  coInstanceRenderer.cpp
  coSphere.cpp
  ProgramCache.cpp
  coLOD.cpp
  coSensor.cpp
  GenericGuiObject.cpp
  MultiChannelDrawer.cpp
  StaticSequence.cpp
  CudaGraphicsResource.cpp
  CudaTextureRectangle.cpp
)

ADD_COVISE_LIBRARY(coOpenPluginUtil SHARED )
TARGET_LINK_LIBRARIES(coOpenPluginUtil coOpenCOVER coOpenVRUI coOSGVRUI
   ${COVISE_GRMSG_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES})

COVISE_WNOERROR(coOpenPluginUtil BASEARCH zebu)

TARGET_LINK_LIBRARIES(coOpenPluginUtil ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(coOpenPluginUtil)
COVISE_INSTALL_HEADERS(PluginUtil ${PLUGINUTIL_HEADERS})
qt_use_modules(coOpenPluginUtil Core)
