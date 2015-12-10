# OpenCOVER kernel (coOpenCOVER)

USING(VTK optional)

USING(MPI optional)
set(COVERKERNEL_TARGET coOpenCOVER)
set(COVERDIR "${COVISEDIR}/src/OpenCOVER")

USING(OpenGL XERCESC)
USING(CG optional)
USING(BOOST)
USING(VRPN optional)

IF(NOT WIN32)
  ADD_DEFINITIONS(-D_OLD_TERMIOS)
ENDIF(NOT WIN32)

INCLUDE_DIRECTORIES(
  ${XERCESC_INCLUDE_DIR}
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

add_subdirectory(input/dev/dtrack) 
add_subdirectory(input/dev/mousebuttons) 
#don't build legacy tracking system, port it to the new system if you need anything
#add_subdirectory(input/dev/legacy)
add_subdirectory(input/dev/vrc) 
add_subdirectory(input/dev/rift) 
add_subdirectory(input/dev/mike)
add_subdirectory(input/dev/vrpn)
add_subdirectory(input/dev/cavelib)
IF(WIN32)
  add_subdirectory(input/dev/windowsHID) 
ENDIF(WIN32)

SET(HEADERS
  ARToolKit.h
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
  VRPinboard.h
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

SET(SOURCES
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
  VRPinboard.cpp
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
  PluginMenu.cpp
  coTranslator.cpp
  VRRegisterSceneGraph.cpp
  MSEventHandler.cpp
  #coVRDePee.cpp
  #coVRDePeePass.cpp
)

SET(MUI_SOURCES
  mui/Element.cpp
  mui/Container.cpp
  mui/Frame.cpp
  mui/LabelElement.cpp
  mui/ValueRegulator.cpp
  mui/Tab.cpp
  mui/TabFolder.cpp
  mui/ToggleButton.cpp
  mui/support/ConfigManager.cpp
  mui/support/ConfigParser.cpp
  mui/support/DefaultValues.cpp
  mui/support/ElementManager.cpp
  mui/support/PositionManager.cpp
  mui/support/EventListener.cpp
)

SET(MUI_HEADERS
  mui/Element.h
  mui/Container.h
  mui/Frame.h
  mui/LabelElement.h
  mui/ValueRegulator.h
  mui/Tab.h
  mui/TabFolder.h
  mui/ToggleButton.h
  mui/support/ConfigManager.h
  mui/support/ConfigParser.h
  mui/support/DefaultValues.h
  mui/support/ElementManager.h
  mui/support/PositionManager.h
  mui/support/EventListener.h
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
  LIST(APPEND SOURCES rel_mcast.cpp)
  LIST(APPEND HEADERS rel_mcast.h)
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

SET(ALL_SOURCES )
SET(ALL_SOURCES_SET SOURCES HEADERS MUI_SOURCES MUI_HEADERS DEVICE_SOURCES DEVICE_HEADERS)

FOREACH(source_set ${ALL_SOURCES_SET})
   FOREACH(tmp_src ${${source_set}})
      LIST(APPEND ALL_SOURCES "${COVERDIR}/cover/${tmp_src}")
   ENDFOREACH(tmp_src)
ENDFOREACH(source_set)

UNSET(SOURCES)
UNSET(HEADERS)

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
endif()

IF(BASEARCHSUFFIX STREQUAL "angus" OR
   BASEARCHSUFFIX STREQUAL "vista")
  IF(MSVC AND QT_VERSION_MAJOR EQUAL 4)
    ADD_COVISE_COMPILE_FLAGS(${COVERKERNEL_TARGET} "/Zc:wchar_t-")
  ENDIF()
ENDIF()
if(NOT APPLE AND NOT WIN32)
      set(EXTRA_LIBS ${EXTRA_LIBS} -ldl)
endif()

IF(WIN32)
  set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)

COVISE_WNOERROR(${COVERKERNEL_TARGET} BASEARCH basilisk tuatara)

TARGET_LINK_LIBRARIES(${COVERKERNEL_TARGET} vtrans coOpenVRUI coOSGVRUI
   ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_GRMSG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${COVISE_GPU_LIBRARIES} ${COVISE_NET_LIBRARY}
   ${XERCESC_LIBRARIES} ${OPENSCENEGRAPH_LIBRARIES} ${TIFF_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(${COVERKERNEL_TARGET})
qt_use_modules(${COVERKERNEL_TARGET} Core Network)
