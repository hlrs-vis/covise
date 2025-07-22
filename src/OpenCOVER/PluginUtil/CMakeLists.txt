# OpenCOVER PluginUtil (coOpenPluginUtil)
USING(BOOST)
USING(CG optional)
USING(GLEW)

IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()

PROJECT(OpenCOVER CXX C)
ADD_DEFINITIONS(-DCOVISE_PLUGIN_UTIL)



if(COVISE_USE_CUDA)
   if(CUDA_FOUND)
      include_directories(${CUDA_INCLUDE_DIRS})
      set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
   endif()
endif()

SET(PLUGINUTIL_HEADERS
  BoxSelection.h
  BoxSelectionInteractor.h
  coArrow.h
  coInstanceRenderer.h
  coLine.h
  coLOD.h
  colors/coColorBar.h
  colors/coColorHUD.h
  colors/coColorMap.h
  colors/ColorBar.h
  colors/ColorMaterials.h
  coPlane.h
  coSensor.h
  coShaderUtil.h
  coSphere.h
  coVR1DTransInteractor.h
  coVR2DTransInteractor.h
  coVR3DGizmo.h
  coVR3DGizmoType.h
  coVR3DRotCenterInteractor.h
  coVR3DRotGizmo.h
  coVR3DRotInteractor.h
  coVR3DScaleGizmo.h
  coVR3DTransGizmo.h
  coVR3DTransInteractor.h
  coVR3DTransRotInteractor.h
  CudaGraphicsResource.h
  CudaTextureRectangle.h
  FeedbackManager.h
  FileReference.h
  GenericGuiObject.h
  ModuleFeedbackManager.h
  ModuleFeedbackPlugin.h
  MultiChannelDrawer.h
  PluginMessageTypes.h
  ProgramCache.h
  SimReference.h
  StaticSequence.h
  Tipsify.h
)

SET(SOURCES
  BoxSelection.cpp
  BoxSelectionInteractor.cpp
  coArrow.cpp
  colors/coColorBar.cpp
  colors/coColorHUD.cpp
  colors/coColorMap.cpp
  colors/ColorBar.cpp
  colors/ColorMaterials.cpp
  coInstanceRenderer.cpp
  coLine.cpp
  coLOD.cpp
  coPlane.cpp
  coSensor.cpp
  coShaderUtil.cpp
  coSphere.cpp
  coVR1DTransInteractor.cpp
  coVR2DTransInteractor.cpp
  coVR3DGizmo.cpp
  coVR3DGizmoType.cpp
  coVR3DRotCenterInteractor.cpp
  coVR3DRotGizmo.cpp
  coVR3DRotInteractor.cpp
  coVR3DScaleGizmo.cpp
  coVR3DTransGizmo.cpp
  coVR3DTransInteractor.cpp
  coVR3DTransRotInteractor.cpp
  CudaGraphicsResource.cpp
  CudaTextureRectangle.cpp
  FeedbackManager.cpp
  FileReference.cpp
  GenericGuiObject.cpp
  ModuleFeedbackManager.cpp
  ModuleFeedbackPlugin.cpp
  ModuleInteraction.cpp
  MultiChannelDrawer.cpp
  ProgramCache.cpp
  SimReference.cpp
  StaticSequence.cpp
)



ADD_COVISE_LIBRARY(coOpenPluginUtil SHARED )
TARGET_LINK_LIBRARIES(coOpenPluginUtil coOpenCOVER coOpenVRUI coOSGVRUI
   ${COVISE_GRMSG_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES})

COVISE_WNOERROR(coOpenPluginUtil BASEARCH zebu)
TARGET_INCLUDE_DIRECTORIES(coOpenPluginUtil PUBLIC ${OPENSCENEGRAPH_INCLUDE_DIRS})




TARGET_LINK_LIBRARIES(coOpenPluginUtil ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(coOpenPluginUtil)
COVISE_INSTALL_HEADERS(PluginUtil ${PLUGINUTIL_HEADERS})
