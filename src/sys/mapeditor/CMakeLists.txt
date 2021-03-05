# @file
# 
# CMakeLists.txt for sys mapeditor
#
# @author Blasius Czink

if(APPLE)
   set(TARGETNAME COVISE)
else()
   set(TARGETNAME mapeditor)
endif()

USING(OpenGL)

ADD_DEFINITIONS(-DTABLET_PLUGIN)

SET(TABLETUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tabletUI")

INCLUDE_DIRECTORIES(
  "${COVISEDIR}/src/OpenCOVER/cover"
  "${TABLETUI_DIR}"
)
  INCLUDE_DIRECTORIES(
    covise
  )

SET(MAPEDITOR_SOURCES
  widgets/MEUserInterface.cpp
  widgets/MEGraphicsView.cpp
  widgets/MERegistry.cpp
  widgets/MEPreference.cpp
  widgets/MESessionSettings.cpp
  widgets/MEHelpViewer.cpp
  widgets/MEFavorites.cpp
  widgets/MEModuleTree.cpp
  widgets/MEDialogTools.cpp
  material/MEMaterialChooser.cpp
  material/MEMaterialDisplay.cpp
  color/MEColorChooser.cpp
  color/MEColorSelector.cpp
  color/MEColorMap.cpp
  hosts/MEDaemon.cpp
  hosts/MEHost.cpp
  ports/MEPort.cpp
  ports/MEDataPort.cpp
  ports/MEParameterPort.cpp
  ports/MEScalarPort.cpp
  ports/MESliderPort.cpp
  ports/MEVectorPort.cpp
  ports/MEFloatScalarPort.cpp
  ports/MEFloatVectorPort.cpp
  ports/MEFloatSliderPort.cpp
  ports/MEIntScalarPort.cpp
  ports/MEIntVectorPort.cpp
  ports/MEIntSliderPort.cpp
  ports/MEChoicePort.cpp
  ports/MEStringPort.cpp
  ports/MEBooleanPort.cpp
  ports/MEFileBrowserPort.cpp
  ports/MEParameterAppearance.cpp
  ports/MEExtendedPart.cpp
  ports/MELineEdit.cpp
  ports/MEColorMapPort.cpp
  ports/MEColormapChoicePort.cpp
  ports/MEMaterialPort.cpp
  ports/MEColorPort.cpp
  ports/MESpinBox.cpp
  ports/METimer.cpp
  handler/MEMainHandler.cpp
  handler/MEPortSelectionHandler.cpp
  handler/MENodeListHandler.cpp
  handler/MELinkListHandler.cpp
  handler/MEHostListHandler.cpp
  handler/MEFileBrowserListHandler.cpp
  handler/MEFavoriteListHandler.cpp
  controlPanel/MEControlPanel.cpp
  controlPanel/MEControlParameter.cpp
  controlPanel/MEControlParameterLine.cpp
  modulePanel/MEModuleParameter.cpp
  modulePanel/MEModuleParameterLine.cpp
  modulePanel/MEModulePanel.cpp
  dataObjects/MEDataTree.cpp
  dataObjects/MEDataViewer.cpp
  dataObjects/MEDataObject.cpp
  dataObjects/MEDataArray.cpp
  nodes/MENode.cpp
  nodes/MENodeLink.cpp
  nodes/MECategory.cpp
  gridProxy/METable.cpp
  gridProxy/MEGridProxy.cpp
  MEApplication.cpp
  MEMain.cpp
)

SET(MAPEDITOR_SOURCES ${MAPEDITOR_SOURCES}
  covise/MEFileBrowser.cpp
  covise/MEMessageHandler.cpp
  covise/MERemotePartner.cpp
  )


SET(MAPEDITOR_HEADERS
  ports/METimer.h
  hosts/MEHost.h
  nodes/MENodeLink.h
  nodes/MECategory.h
)

SET(MAPEDITOR_MOC_HEADERS
  color/MEColorChooser.h
  color/MEColorSelector.h
  color/MEColorMap.h
  widgets/MEUserInterface.h
  widgets/MEGraphicsView.h
  widgets/MERegistry.h
  widgets/MEPreference.h
  widgets/MESessionSettings.h
  widgets/MEHelpViewer.h
  widgets/MEFavorites.h
  widgets/MEModuleTree.h
  widgets/MEDialogTools.h
  ports/MEPort.h
  ports/MEDataPort.h
  ports/MEParameterPort.h
  ports/MEScalarPort.h
  ports/MESliderPort.h
  ports/MEVectorPort.h
  ports/MEFloatScalarPort.h
  ports/MEFloatVectorPort.h
  ports/MEFloatSliderPort.h
  ports/MEIntScalarPort.h
  ports/MEIntVectorPort.h
  ports/MEIntSliderPort.h
  ports/MEChoicePort.h
  ports/MEStringPort.h
  ports/MEColorMapPort.h
  ports/MEColormapChoicePort.h
  ports/MEMaterialPort.h
  ports/MEColorPort.h
  ports/MEBooleanPort.h
  ports/MEFileBrowserPort.h
  ports/MEParameterAppearance.h
  ports/MEExtendedPart.h
  ports/MELineEdit.h
  ports/MESpinBox.h
  material/MEMaterialDisplay.h
  material/MEMaterialChooser.h
  controlPanel/MEControlPanel.h
  controlPanel/MEControlParameter.h
  controlPanel/MEControlParameterLine.h
  handler/MEMainHandler.h
  handler/MEPortSelectionHandler.h
  handler/MENodeListHandler.h
  handler/MELinkListHandler.h
  handler/MEHostListHandler.h
  handler/MEFileBrowserListHandler.h
  handler/MEFavoriteListHandler.h
  modulePanel/MEModuleParameter.h
  modulePanel/MEModuleParameterLine.h
  modulePanel/MEModulePanel.h
  dataObjects/MEDataTree.h
  dataObjects/MEDataViewer.h
  dataObjects/MEDataObject.h
  dataObjects/MEDataArray.h
  gridProxy/METable.h
  gridProxy/MEGridProxy.h
  hosts/MEDaemon.h
  nodes/MENode.h
  MEApplication.h
)

SET(MAPEDITOR_MOC_HEADERS ${MAPEDITOR_MOC_HEADERS}
  covise/MEFileBrowser.h
  covise/MEMessageHandler.h
  covise/MERemotePartner.h)

if(COVISE_USE_QT4)
   SET(MAPEDITOR_MOC_HEADERS ${MAPEDITOR_MOC_HEADERS}
      widgets/METextBrowser.h
   )
   SET(MAPEDITOR_SOURCES ${MAPEDITOR_SOURCES}
      widgets/METextBrowser.cpp
   )
   ADD_DEFINITIONS(-DUSE_TEXTBROWSER)
endif()

if (COVISE_USE_WEBENGINE)
   SET(MAPEDITOR_MOC_HEADERS ${MAPEDITOR_MOC_HEADERS}
     widgets/MEWebEnginePage.h
   )
   SET(MAPEDITOR_SOURCES ${MAPEDITOR_SOURCES}
     widgets/MEWebEnginePage.cpp
   )
endif()

# BEGIN: TUI Stuff
INCLUDE("${TABLETUI_DIR}/tabletUI.cmake")

# prepend proper path to TUI_SOURCES & TUI_HEADERS
SET(ME_TUI_SOURCES )
FOREACH(tmp_src ${TUI_SOURCES})
  LIST(APPEND ME_TUI_SOURCES "${TABLETUI_DIR}/${tmp_src}")
ENDFOREACH(tmp_src)

SET(ME_TUI_HEADERS )
FOREACH(tmp_src ${TUI_HEADERS})
  LIST(APPEND ME_TUI_HEADERS "${TABLETUI_DIR}/${tmp_src}")
ENDFOREACH(tmp_src)

SET(ME_TUI_MOC_HEADERS )
FOREACH(tmp_src ${TUI_MOC_HEADERS})
  LIST(APPEND ME_TUI_MOC_HEADERS "${TABLETUI_DIR}/${tmp_src}")
ENDFOREACH(tmp_src)

QT_WRAP_CPP(ME_TUI_MOC_SRCS ${ME_TUI_MOC_HEADERS} OPTIONS -DTABLET_PLUGIN)
# END: TUI Stuff

QT_ADD_RESOURCES(MAPEDITOR_RCC_FILES
  symbols.qrc
  certs.qrc
  icons.qrc
  about.qrc
  
)

QT_WRAP_CPP(MAPEDITOR_MOC_SRCS ${MAPEDITOR_MOC_HEADERS})

QT_WRAP_UI(MAPEDITOR_UI_SRCS "forms/MEAboutDialog.ui" "forms/MERemotePartner.ui")
QT_WRAP_UI(widgets/MERemotePartner.ui)
ADD_COVISE_EXECUTABLE(${TARGETNAME} MACOSX_BUNDLE
  ${MAPEDITOR_SOURCES} ${MAPEDITOR_HEADERS}
  ${MAPEDITOR_MOC_HEADERS} ${MAPEDITOR_MOC_SRCS}
  ${MAPEDITOR_RCC_FILES} ${MAPEDITOR_UI_SRCS}
  ${ME_TUI_SOURCES} ${ME_TUI_HEADERS} ${ME_TUI_MOC_SRCS} ${ME_TUI_MOC_HEADERS}
)
qt_use_modules(${TARGETNAME} Widgets Xml Network OpenGL Svg UiTools PrintSupport Quick)
if (COVISE_USE_WEBENGINE)
  qt_use_modules(${TARGETNAME} WebEngineWidgets)
  ADD_DEFINITIONS(-DUSE_WEBENGINE)
elseif (COVISE_USE_WEBKIT)
  ADD_DEFINITIONS(-DUSE_WEBKIT)
  qt_use_modules(${TARGETNAME} WebKit WebKitWidgets)
endif()
if(Qt5Positioning_FOUND)
  qt_use_modules(${TARGETNAME} Positioning Location)
endif(Qt5Positioning_FOUND)
if(COVISE_USE_QT4)
   IF(QT_QTWEBKIT_FOUND)
      ADD_DEFINITIONS(-DQT_WEBKIT)
      SET(QT_USE_QTWEBKIT 1)
   else()
      MESSAGE("Not building support for downloading docs from web server: no QtWebKit found")
   ENDIF()
endif()

IF(APPLE)
   SET_TARGET_PROPERTIES(${TARGETNAME} PROPERTIES OUTPUT_NAME "COVISE")
   ADD_COVISE_COMPILE_FLAGS(${TARGETNAME} "-Wno-deprecated")
ENDIF(APPLE)

IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   ADD_COVISE_COMPILE_FLAGS(${TARGETNAME} "-Wno-error=deprecated-declarations")
ENDIF()

IF(MSVC AND QT_VERSION_MAJOR EQUAL 4)
   ADD_COVISE_COMPILE_FLAGS(${TARGETNAME} "/Zc:wchar_t-")
ENDIF()

TARGET_LINK_LIBRARIES(${TARGETNAME} coDo coCore coConfig coUtil coMessages ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(${TARGETNAME})
