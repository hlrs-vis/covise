# OpenCOVER kernel (coOpenCOVER)

USING(MPI optional)
set(COVERKERNEL_TARGET coOpenCOVER)
set(COVERDIR "${COVISEDIR}/src/OpenCOVER")

USING(OpenGL XERCESC)
USING(GLEW)
USING(BOOST)

IF(NOT WIN32)
  ADD_DEFINITIONS(-D_OLD_TERMIOS)
ENDIF(NOT WIN32)

INCLUDE_DIRECTORIES(
  ${XERCESC_INCLUDE_DIR}
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(COVER_HEADERS
  ARToolKit.h
  Deletable.h
  coActionUserData.h
  coBillboard.h
  coClusterStat.h
  coCollabInterface.h
  coCommandLine.h
  coCoverConfig.h
  coCullVisitor.h
  coOnscreenDebug.h
  coHud.h
  coInteractor.h
  coIntersection.h
  coIntersectionUtil.h
  coPartnerMenuItem.h
  coVrbRegistryAccess.h
  coVRCommunication.h
  coVRConfig.h
  coVRDynLib.h
  coVRLabel.h
  coVRPlugin.h
  coVRPluginList.h
  coVRPluginSupport.h
  coVRFileManager.h
  coVRIOBase.h
  coVRIOReader.h
  coVRMSController.h
  coVRPartner.h
  coVRSlave.h
  coVRShader.h
  coVRStatsDisplay.h
  coVRTui.h
  coVRSceneView.h
  coVRTouchTable.h
  EnableGLDebugOperation.h
  OpenCOVER.h
  RenderObject.h
  VRAvatar.h
  QuitDialog.h
  VRSceneGraph.h
  coVRLighting.h
  coVRAnimationManager.h
  coVRNavigationManager.h
  coVRCollaboration.h
  coVRSelectionManager.h
  coVRIntersectionInteractorManager.h
  coVRIntersectionInteractor.h
  coVRShadowManager.h
  VRViewer.h
  coVRRenderer.h
  VRVruiButtons.h
  VRVruiRenderInterface.h
  VRWindow.h
  coShutDownHandler.h
  PluginMenu.h
  EventReceiver.h
  coMeasurement.h
  coTranslator.h
  VRRegisterSceneGraph.h
  MSEventHandler.h
  tridelity.h
  # tui file browser headers
  coTUIFileBrowser/AGData.h
  coTUIFileBrowser/FileSysAccess.h
  coTUIFileBrowser/IData.h
  coTUIFileBrowser/LocalData.h
  coTUIFileBrowser/NetHelp.h
  coTUIFileBrowser/VRBData.h
  coTUIFileBrowser/IRemoteData.h
  #coVRDePee.h
  #coVRDePeePass.h
)

SET(COVER_SOURCES
  Deletable.cpp
  VRWindow.cpp
  coCommandLine.cpp
  coCoverConfig.cpp
  coVRConfig.cpp
  coVRPlugin.cpp
  coVRPluginList.cpp
  coVRPluginSupport.cpp
  coVRFileManager.cpp
  coVRIOBase.cpp
  coVRIOReader.cpp
  coVRShader.cpp
  coVRDynLib.cpp
  coInteractor.cpp
  coClusterStat.cpp
  coVrbRegistryAccess.cpp
  coVRCommunication.cpp
  coVRPartner.cpp
  coVRSlave.cpp
  coVRMSController.cpp
  coVRSceneView.cpp
  coVRStatsDisplay.cpp
  coVRTouchTable.cpp
  EnableGLDebugOperation.cpp
  OpenCOVER.cpp
  QuitDialog.cpp
  VRAvatar.cpp
  coCullVisitor.cpp
  VRSceneGraph.cpp
  coVRLighting.cpp
  coVRAnimationManager.cpp
  coVRShadowManager.cpp
  ARToolKit.cpp
  coBillboard.cpp
  VRViewer.cpp
  coVRRenderer.cpp
  VRVruiRenderInterface.cpp
  VRVruiButtons.cpp
  coVRTui.cpp
  coActionUserData.cpp
  coIntersection.cpp
  coIntersectionUtil.cpp
  coPartnerMenuItem.cpp
  coCollabInterface.cpp
  coVRLabel.cpp
  coHud.cpp
  coOnscreenDebug.cpp
  coTabletUI.cpp
  coTUIListener.cpp
  RenderObject.cpp
  coVRNavigationManager.cpp
  coVRCollaboration.cpp
  coVRSelectionManager.cpp
  coVRIntersectionInteractorManager.cpp
  coVRIntersectionInteractor.cpp
  EventReceiver.cpp
  coMeasurement.cpp
  coTUIFileBrowser/AGData.cpp
  coTUIFileBrowser/FileSysAccess.cpp
  coTUIFileBrowser/IData.cpp
  coTUIFileBrowser/LocalData.cpp
  coTUIFileBrowser/NetHelp.cpp
  coTUIFileBrowser/VRBData.cpp
  coShutDownHandler.cpp
  coTranslator.cpp
  PluginMenu.cpp
  VRRegisterSceneGraph.cpp
  MSEventHandler.cpp
  tridelity.cpp
  #coVRDePee.cpp
  #coVRDePeePass.cpp
)

SET(UI_HEADERS
    ui/Export.h
    ui/Action.h
    ui/Button.h
    ui/Container.h
    ui/Element.h
    ui/Group.h
    ui/Label.h
    ui/Manager.h
    ui/Menu.h
    ui/Owner.h
    ui/ButtonGroup.h
    ui/SelectionList.h
    ui/ShortcutListener.h
    ui/Slider.h
    ui/View.h
    ui/VruiView.h
)

SET(UI_SOURCES
    ui/Action.cpp
    ui/Button.cpp
    ui/Container.cpp
    ui/Element.cpp
    ui/Group.cpp
    ui/Label.cpp
    ui/Manager.cpp
    ui/Menu.cpp
    ui/Owner.cpp
    ui/ButtonGroup.cpp
    ui/SelectionList.cpp
    ui/ShortcutListener.cpp
    ui/Slider.cpp
    ui/View.cpp
    ui/VruiView.cpp
)

SET(DEVICE_SOURCES
   input/input.cpp
   input/inputdevice.cpp
   input/input_const.cpp
   input/person.cpp
   input/inputsource.cpp
   input/trackingbody.cpp
   input/buttondevice.cpp
   input/valuator.cpp

   input/coMousePointer.cpp
)

SET(DEVICE_HEADERS
   input/input.h
   input/inputdevice.h
   input/input_const.h
   input/person.h
   input/inputsource.h
   input/trackingbody.h
   input/buttondevice.h
   input/valuator.h

   input/VRKeys.h
   input/coMousePointer.h
)

IF(HAVE_NORM) # or NORM_FOUND ... TODO
    LIST(APPEND COVER_SOURCES rel_mcast.cpp)
  LIST(APPEND COVER_HEADERS rel_mcast.h)
ENDIF(HAVE_NORM)

SET(COVER_MOCS
  coTabletUI.h
)

SET(TMP_COVER_MOCS )
FOREACH(tmp_src ${COVER_MOCS})
   LIST(APPEND TMP_COVER_MOCS "${COVERDIR}/cover/${tmp_src}")
ENDFOREACH(tmp_src)
SET(COVER_MOCS ${TMP_COVER_MOCS})

QT_WRAP_CPP(MOCS ${COVER_MOCS})

SET(ALL_HEADERS ${COVER_HEADERS} ${DEVICE_HEADERS} ${UI_HEADERS})
SET(ALL_SOURCES ${COVER_SOURCES} ${DEVICE_SOURCES} ${UI_SOURCES})

ADD_COVISE_LIBRARY(${COVERKERNEL_TARGET} SHARED ${ALL_SOURCES} ${MOCS})

COVISE_USE_OPENMP(${COVERKERNEL_TARGET})

IF(BASEARCHSUFFIX STREQUAL "edgy" OR
   BASEARCHSUFFIX STREQUAL "eft" OR
   BASEARCHSUFFIX STREQUAL "bishorn" OR
   BASEARCHSUFFIX STREQUAL "basilisk")
  ADD_COVISE_COMPILE_FLAGS(${COVERKERNEL_TARGET} "-fno-strict-aliasing")
ENDIF()

if(APPLE)
   add_covise_compile_flags(${COVERKERNEL_TARGET} "-Wno-deprecated")
   add_covise_compile_flags(${COVERKERNEL_TARGET} "-Wno-error=null-dereference")
endif()

IF(BASEARCHSUFFIX STREQUAL "angus" OR
   BASEARCHSUFFIX STREQUAL "vista")
  IF(MSVC AND QT_VERSION_MAJOR EQUAL 4)
    ADD_COVISE_COMPILE_FLAGS(${COVERKERNEL_TARGET} "/Zc:wchar_t-")
  ENDIF()
ENDIF()
IF (UNIX)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${CMAKE_DL_LIBS})
ENDIF()

IF(WIN32)
  set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)

COVISE_WNOERROR(${COVERKERNEL_TARGET} BASEARCH basilisk tuatara zebu)

TARGET_LINK_LIBRARIES(${COVERKERNEL_TARGET} coOpenVRUI coOSGVRUI
   ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_GRMSG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${COVISE_GPU_LIBRARIES} ${COVISE_NET_LIBRARY}
   ${XERCESC_LIBRARIES} ${OPENSCENEGRAPH_LIBRARIES} ${TIFF_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(${COVERKERNEL_TARGET})
COVISE_INSTALL_HEADERS(cover ${COVER_HEADERS} ${COVER_MOCS})
COVISE_INSTALL_HEADERS(cover/ui ${UI_HEADERS})
COVISE_INSTALL_HEADERS(cover/input ${DEVICE_HEADERS})
qt_use_modules(${COVERKERNEL_TARGET} Core Network)
