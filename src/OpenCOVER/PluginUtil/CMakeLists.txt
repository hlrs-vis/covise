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
  BoxSelection.h
  BoxSelectionInteractor.h
  ColorBar.h
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
  coArrow.h
  coColorBar.h
  coColorMap.h
  coInstanceRenderer.h
  coLOD.h
  coLine.h
  coPlane.h
  coSensor.h
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
)

SET(SOURCES
  BoxSelection.cpp
  BoxSelectionInteractor.cpp
  ColorBar.cpp
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
  coArrow.cpp
  coColorBar.cpp
  coColorMap.cpp
  coInstanceRenderer.cpp
  coLOD.cpp
  coLine.cpp
  coPlane.cpp
  coSensor.cpp
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
)

if(open62541_FOUND)
  SET(HEADERS
      ${HEADERS}
      OpcUaClient/opcua.h
      OpcUaClient/opcuaClient.h
  )

  SET(SOURCES
    ${SOURCES}    
      OpcUaClient/opcua.cpp
      OpcUaClient/opcuaClient.cpp
  )
endif()

ADD_COVISE_LIBRARY(coOpenPluginUtil SHARED )
TARGET_LINK_LIBRARIES(coOpenPluginUtil coOpenCOVER coOpenVRUI coOSGVRUI
   ${COVISE_GRMSG_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES})

COVISE_WNOERROR(coOpenPluginUtil BASEARCH zebu)
if(open62541_FOUND)
  TARGET_LINK_LIBRARIES(coOpenPluginUtil open62541::open62541)
endif()


TARGET_LINK_LIBRARIES(coOpenPluginUtil ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(coOpenPluginUtil)
COVISE_INSTALL_HEADERS(PluginUtil ${PLUGINUTIL_HEADERS})
qt_use_modules(coOpenPluginUtil Core)

