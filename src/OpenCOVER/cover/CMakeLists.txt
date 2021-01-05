# OpenCOVER kernel (coOpenCOVER)

USING(MPI optional)
set(COVERKERNEL_TARGET coOpenCOVER)
set(COVERDIR "${COVISEDIR}/src/OpenCOVER")

USING(OpenGL)
USING(XERCESC)
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
  # tui file browser headers
  #coVRDePee.h
  #coVRDePeePass.h
  ARToolKit.h
  Deletable.h
  EventReceiver.h
  InitGLOperation.h
  MSEventHandler.h
  MatrixSerializer.h
  OpenCOVER.h
  PluginMenu.h
  QuitDialog.h
  RenderObject.h
  VRAvatar.h
  VRRegisterSceneGraph.h
  VRSceneGraph.h
  VRViewer.h
  VRVruiButtons.h
  VRVruiRenderInterface.h
  VRWindow.h
  VruiPartnerMenuItem.h
  coActionUserData.h
  coBillboard.h
  coClusterStat.h
  coCollabInterface.h
  coCommandLine.h
  coCoverConfig.h
  coCullVisitor.h
  coHud.h
  coInteractor.h
  coIntersection.h
  coMeasurement.h
  coOnscreenDebug.h
  coShutDownHandler.h
  coTUIFileBrowser/AGData.h
  coTUIFileBrowser/IData.h
  coTUIFileBrowser/IRemoteData.h
  coTUIFileBrowser/LocalData.h
  coTUIFileBrowser/VRBData.h
  coTranslator.h
  coVRAnimationManager.h
  coVRCollaboration.h
  coVRCommunication.h
  coVRConfig.h
  coVRDynLib.h
  coVRFileManager.h
  coVRIOBase.h
  coVRIOReader.h
  coVRIntersectionInteractor.h
  coVRIntersectionInteractorManager.h
  coVRLabel.h
  coVRLighting.h
  coVRMSController.h
  coVRMessageSender.h
  coVRNavigationManager.h
  coVRPartner.h
  coVRPlugin.h
  coVRPluginList.h
  coVRPluginSupport.h
  coVRRenderer.h
  coVRSceneView.h
  coVRSelectionManager.h
  coVRShader.h
  coVRShadowManager.h
  coVRSlave.h
  coVRStatsDisplay.h
  coVRTouchTable.h
  coVRTui.h
  coVrbMenu.h
  tridelity.h
)

SET(COVER_SOURCES
  ARToolKit.cpp
  Deletable.cpp
  EventReceiver.cpp
  InitGLOperation.cpp
  MSEventHandler.cpp
  MatrixSerializer.cpp
  OpenCOVER.cpp
  PluginMenu.cpp
  QuitDialog.cpp
  RenderObject.cpp
  VRAvatar.cpp
  VRRegisterSceneGraph.cpp
  VRSceneGraph.cpp
  VRViewer.cpp
  VRVruiButtons.cpp
  VRVruiRenderInterface.cpp
  VRWindow.cpp
  VruiPartnerMenuItem.cpp
  coActionUserData.cpp
  coBillboard.cpp
  coClusterStat.cpp
  coCollabInterface.cpp
  coCommandLine.cpp
  coCoverConfig.cpp
  coCullVisitor.cpp
  coHud.cpp
  coInteractor.cpp
  coIntersection.cpp
  coMeasurement.cpp
  coOnscreenDebug.cpp
  coShutDownHandler.cpp
  coTUIFileBrowser/AGData.cpp
  coTUIFileBrowser/IData.cpp
  coTUIFileBrowser/LocalData.cpp
  coTUIFileBrowser/VRBData.cpp
  coTUIListener.cpp
  coTUISGBrowserTab.cpp
  coTabletUI.cpp
  coTranslator.cpp
  coVRAnimationManager.cpp
  coVRCollaboration.cpp
  coVRCommunication.cpp
  coVRConfig.cpp
  coVRDynLib.cpp
  coVRFileManager.cpp
  coVRIOBase.cpp
  coVRIOReader.cpp
  coVRIntersectionInteractor.cpp
  coVRIntersectionInteractorManager.cpp
  coVRLabel.cpp
  coVRLighting.cpp
  coVRMSController.cpp
  coVRMessageSender.cpp
  coVRNavigationManager.cpp
  coVRPartner.cpp
  coVRPlugin.cpp
  coVRPluginList.cpp
  coVRPluginSupport.cpp
  coVRRenderer.cpp
  coVRSceneView.cpp
  coVRSelectionManager.cpp
  coVRShader.cpp
  coVRShadowManager.cpp
  coVRSlave.cpp
  coVRStatsDisplay.cpp
  coVRTouchTable.cpp
  coVRTui.cpp
  coVrbMenu.cpp
  tridelity.cpp
  #coVRDePee.cpp
  #coVRDePeePass.cpp
)

SET(UI_HEADERS
    ui/Export.h
    ui/Action.h
    ui/Button.h
    ui/CollaborativePartner.h
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
    ui/TextField.h
    ui/EditField.h
    ui/FileBrowser.h
    ui/SpecialElement.h
    ui/View.h
    ui/VruiView.h
    ui/TabletView.h
)

SET(UI_SOURCES
    ui/Action.cpp
    ui/Button.cpp
    ui/CollaborativePartner.cpp
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
    ui/TextField.cpp
    ui/EditField.cpp
    ui/FileBrowser.cpp
    ui/SpecialElement.cpp
    ui/View.cpp
    ui/VruiView.cpp
    ui/TabletView.cpp
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
  coVrbMenu.h
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

TARGET_LINK_LIBRARIES(${COVERKERNEL_TARGET} coOpenVRUI coOSGVRUI coQtUtil coMessages
   ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_GRMSG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${COVISE_GPU_LIBRARIES} ${COVISE_NET_LIBRARY}
   ${XERCESC_LIBRARIES} ${OPENSCENEGRAPH_LIBRARIES} ${TIFF_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(${COVERKERNEL_TARGET})
COVISE_INSTALL_HEADERS(cover ${COVER_HEADERS} ${COVER_MOCS})
COVISE_INSTALL_HEADERS(cover/ui ${UI_HEADERS})
COVISE_INSTALL_HEADERS(cover/input ${DEVICE_HEADERS})

qt_use_modules(${COVERKERNEL_TARGET} Core Network)
