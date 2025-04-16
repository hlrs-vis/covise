# VIVE core (coOpenVIVE)

USING(MPI optional)
USING(CURL optional)
set(VIVECORE_TARGET vvCore)
set(VIVEDIR "${COVISEDIR}/src/VIVE")

#USING(XERCESC)
#USING(BOOST)
USING(vsg)

IF(NOT WIN32)
  ADD_DEFINITIONS(-D_OLD_TERMIOS)
ENDIF(NOT WIN32)
ADD_DEFINITIONS(-DCONFIG_NAMESPACE=vive)

INCLUDE_DIRECTORIES(
  ${XERCESC_INCLUDE_DIR}
  ${VIVEDIR}/../OpenCOVER
)

SET(VIVE_HEADERS
  vvActionUserData.h
  vvAnimationManager.h
  vvAvatar.h
  vvBillboard.h
  vvClusterStat.h
  vvCollabInterface.h
  vvCollaboration.h
  vvCommandLine.h
  vvCommunication.h
  vvConfig.h
  vvDeletable.h
  vvDynLib.h
  vvEventReceiver.h
  vvFileManager.h
  vvHud.h
  vvInteractor.h
  vvIntersection.h
  vvIntersectionInteractor.h
  vvIntersectionInteractorManager.h
  vvIOBase.h
  vvIOReader.h
  vvLabel.h
  vvLighting.h
  vvMathUtils.h
  vvMarkerTracking.h
  vvMatrixSerializer.h
  vvMeasurement.h
  vvMessageSender.h
  vvMSController.h
  vvMSEventHandler.h
  vvNavigationManager.h
  vvOnscreenDebug.h
  vvPartner.h
  vvPlugin.h
  vvPluginList.h
  vvPluginMenu.h
  vvPluginSupport.h
  vvQuitDialog.h
  vvRegisterSceneGraph.h
  vvRenderObject.h
  vvSceneGraph.h
  vvSelectionManager.h
  vvShader.h
  vvShutDownHandler.h
  vvSidecarConfigBridge.h
  vvSlave.h
  vvTabletUI.h
  vvTabletCovConfig.h
  vvTouchTable.h
  vvTranslator.h
  vvTui.h
  vvTUIListener.h
  vvViewer.h
  vvVIVE.h
  vvVIVEConfig.h
  vvVrbMenu.h
  vvVruiButtons.h
  vvVruiPartnerMenuItem.h
  vvVruiRenderInterface.h
  vvWindows.h
  units.h
)

SET(VIVE_SOURCES
  vvActionUserData.cpp
  vvAnimationManager.cpp
  vvAvatar.cpp
  vvBillboard.cpp
  vvClusterStat.h
  vvCollabInterface.cpp
  vvCollaboration.cpp
  vvCommandLine.cpp
  vvCommunication.cpp
  vvConfig.cpp
  vvDeletable.cpp
  vvDynLib.cpp
  vvEventReceiver.cpp
  vvFileManager.cpp
  vvHud.cpp
  vvInteractor.cpp
  vvIntersection.cpp
  vvIntersectionInteractor.cpp
  vvIntersectionInteractorManager.cpp
  vvIOBase.cpp
  vvIOReader.cpp
  vvLabel.cpp
  vvLighting.cpp
  vvMarkerTracking.cpp
  vvMatrixSerializer.cpp
  vvMeasurement.cpp
  vvMessageSender.cpp
  vvMSController.cpp
  vvMSEventHandler.cpp
  vvNavigationManager.cpp
  vvOnscreenDebug.cpp
  vvPartner.cpp
  vvPlugin.cpp
  vvPluginList.cpp
  vvPluginMenu.cpp
  vvPluginSupport.cpp
  vvQuitDialog.cpp
  vvRegisterSceneGraph.cpp
  vvRenderObject.cpp
  vvSceneGraph.cpp
  vvSelectionManager.cpp
  vvShader.cpp
  vvShutDownHandler.cpp
  vvSidecarConfigBridge.cpp
  vvSlave.cpp
  vvTabletUI.cpp
  vvTabletCovConfig.cpp
  vvTouchTable.cpp
  vvTranslator.cpp
  vvTui.cpp
  vvTUIListener.cpp
  vvTUISGBrowserTab.cpp
  vvViewer.cpp
  vvVIVE.cpp
  vvVIVEConfig.cpp
  vvVrbMenu.cpp
  vvVruiButtons.cpp
  vvVruiPartnerMenuItem.cpp
  vvVruiRenderInterface.cpp
  vvWindows.cpp
  units.cpp
)

SET(UI_HEADERS
    ui/Action.h
    ui/Button.h
    ui/ButtonGroup.h
    ui/CollaborativePartner.h
    ui/Container.h
    ui/CovconfigLink.h
    ui/EditField.h
    ui/Element.h
    ui/Export.h
    ui/FileBrowser.h
    ui/Group.h
    ui/Label.h
    ui/Manager.h
    ui/Menu.h
    ui/Owner.h
    ui/SelectionList.h
    ui/ShortcutListener.h
    ui/Slider.h
    ui/SpecialElement.h
    ui/TabletView.h
    ui/TextField.h
    ui/VectorEditField.h
    ui/View.h
    ui/VruiView.h
)

SET(UI_SOURCES
    ui/Action.cpp
    ui/Button.cpp
    ui/ButtonGroup.cpp
    ui/CollaborativePartner.cpp
    ui/Container.cpp
    ui/CovconfigLink.cpp
    ui/EditField.cpp
    ui/Element.cpp
    ui/FileBrowser.cpp
    ui/Group.cpp
    ui/Label.cpp
    ui/Manager.cpp
    ui/Menu.cpp
    ui/Owner.cpp
    ui/SelectionList.cpp
    ui/ShortcutListener.cpp
    ui/Slider.cpp
    ui/SpecialElement.cpp
    ui/TabletView.cpp
    ui/TextField.cpp
    ui/VectorEditField.cpp    
    ui/View.cpp
    ui/VruiView.cpp
)

SET(DEVICE_SOURCES
   input/input.cpp
   input/inputdevice.cpp
   input/input_const.cpp
   input/gadget.cpp
   input/person.cpp
   input/inputsource.cpp
   input/trackingbody.cpp
   input/buttondevice.cpp
   input/valuator.cpp

   input/vvMousePointer.cpp
   input/deviceDiscovery.cpp
)

SET(DEVICE_HEADERS
   input/input.h
   input/inputdevice.h
   input/input_const.h
   input/gadget.h
   input/person.h
   input/inputsource.h
   input/trackingbody.h
   input/buttondevice.h
   input/valuator.h

   input/VRKeys.h
   input/vvMousePointer.h
   input/deviceDiscovery.h
)

SET(FB_HEADERS
  vvTUIFileBrowser/AGData.h
  vvTUIFileBrowser/IData.h
  vvTUIFileBrowser/IRemoteData.h
  vvTUIFileBrowser/LocalData.h
  vvTUIFileBrowser/VRBData.h
  )

set(FB_SOURCES
  vvTUIFileBrowser/AGData.cpp
  vvTUIFileBrowser/IData.cpp
  vvTUIFileBrowser/LocalData.cpp
  vvTUIFileBrowser/VRBData.cpp
)

IF(HAVE_NORM) # or NORM_FOUND ... TODO
    LIST(APPEND VIVE_SOURCES rel_mcast.cpp)
  LIST(APPEND VIVE_HEADERS rel_mcast.h)
ENDIF(HAVE_NORM)

SET(VIVE_MOCS
  vvTabletUI.h
  vvVrbMenu.h
)

if(COVISE_USE_QT)
SET(TMP_VIVE_MOCS )
FOREACH(tmp_src ${VIVE_MOCS})
   LIST(APPEND TMP_VIVE_MOCS "${VIVEDIR}/core/${tmp_src}")
ENDFOREACH(tmp_src)
SET(VIVE_MOCS ${TMP_VIVE_MOCS})

QT_WRAP_CPP(MOCS ${VIVE_MOCS})
endif()

SET(ALL_HEADERS ${VIVE_HEADERS} ${DEVICE_HEADERS} ${UI_HEADERS})
SET(ALL_SOURCES ${VIVE_SOURCES} ${DEVICE_SOURCES} ${UI_SOURCES})

if(COVISE_USE_QT)
    SET(ALL_HEADERS ${ALL_HEADERS} ${FB_HEADERS})
    SET(ALL_SOURCES ${ALL_SOURCES} ${FB_SOURCES})
endif()

ADD_COVISE_LIBRARY(${VIVECORE_TARGET} SHARED  ${ALL_HEADERS} ${ALL_SOURCES} ${MOCS})
target_include_directories(${VIVECORE_TARGET} PUBLIC ${PTHREAD_INCLUDE_DIR})
COVISE_USE_OPENMP(${VIVECORE_TARGET})

IF(BASEARCHSUFFIX STREQUAL "edgy" OR
   BASEARCHSUFFIX STREQUAL "eft" OR
   BASEARCHSUFFIX STREQUAL "bishorn" OR
   BASEARCHSUFFIX STREQUAL "basilisk")
  ADD_COVISE_COMPILE_FLAGS(${VIVECORE_TARGET} "-fno-strict-aliasing")
ENDIF()

if(APPLE)
   add_covise_compile_flags(${VIVECORE_TARGET} "-Wno-deprecated")
   add_covise_compile_flags(${VIVECORE_TARGET} "-Wno-error=null-dereference")
endif()

IF(BASEARCHSUFFIX STREQUAL "angus" OR
   BASEARCHSUFFIX STREQUAL "vista")
  IF(MSVC AND QT_VERSION_MAJOR EQUAL 4)
    ADD_COVISE_COMPILE_FLAGS(${VIVECORE_TARGET} "/Zc:wchar_t-")
  ENDIF()
ENDIF()
IF (UNIX)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${CMAKE_DL_LIBS})
ENDIF()

IF(WIN32)
  set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)

COVISE_WNOERROR(${VIVECORE_TARGET} BASEARCH basilisk tuatara zebu)


TARGET_LINK_LIBRARIES(${VIVECORE_TARGET} coOpenVRUI coVSGVRUI coMessages vvOpenConfig
   ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_GRMSG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${COVISE_GPU_LIBRARIES} ${COVISE_NET_LIBRARY}
   ${XERCESC_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_LIBS})

IF(CURL_FOUND)
  USING(CURL)
  add_dependencies(${VIVECORE_TARGET} coCurlHTTPClient)
  TARGET_LINK_LIBRARIES(${VIVECORE_TARGET} coCurlHTTPClient)
ENDIF()

target_include_directories(${VIVECORE_TARGET} PRIVATE ${EXTRA_INCS})
COVISE_INSTALL_TARGET(${VIVECORE_TARGET})
COVISE_INSTALL_HEADERS(VIVE ${VIVE_HEADERS} ${VIVE_MOCS})
COVISE_INSTALL_HEADERS(VIVE/ui ${UI_HEADERS})
COVISE_INSTALL_HEADERS(VIVE/input ${DEVICE_HEADERS})

if(COVISE_USE_QT)
    TARGET_LINK_LIBRARIES(${VIVECORE_TARGET} coQtUtil)
    qt_use_modules(${VIVECORE_TARGET} Core Network)
endif()
