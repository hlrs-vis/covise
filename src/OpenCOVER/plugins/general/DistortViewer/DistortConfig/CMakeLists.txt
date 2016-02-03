USING(OSGQT)
USING(BOOST)
USING(VTK optional)
#OpenCOVER plugins general DistortConfig

INCLUDE_DIRECTORIES(..)

SET(HEADERS
  AboutDialog.h
  CylinderGroup.h
  DomeGroup.h
  MainWindow.h
  OpenGeoGroup.h
  PlaneGroup.h
  SceneConf.h
  SettingsDialog.h
  ViewerWidget.h
)

SET(SOURCES
  AboutDialog.cpp
  CylinderGroup.cpp
  DistortConfigMain.cpp
  DomeGroup.cpp
  MainWindow.cpp
  OpenGeoGroup.cpp
  PlaneGroup.cpp
  SceneConf.cpp
  SettingsDialog.cpp
  ViewerWidget.cpp
  ../KoordAxis.cpp
  ../ScreenCylinder.cpp
  ../ScreenDome.cpp
  ../ScreenPlane.cpp
  ../Screen.cpp
  ../ScreenLoad.cpp
  ../Scene.cpp
  ../Projector.cpp
  ../Settings.cpp
  ../XmlTools.cpp
  ../HelpFuncs.cpp
  ../SceneVis.cpp
  ../VisScene.cpp
)
SET(DIST_MOC_HEADERS
  AboutDialog.h
  CylinderGroup.h
  DomeGroup.h
  MainWindow.h
  OpenGeoGroup.h
  PlaneGroup.h
  SettingsDialog.h)
  
QT_ADD_RESOURCES(DIST_RCC_FILES
  DistortConfigurator.qrc
)

QT_WRAP_UI(DIST_UI_SRCS AboutDialog.ui  CylinderGroup.ui  DomeGroup.ui  MainWindow.ui  OpenGeoGroup.ui  PlaneGroup.ui  SettingsDialog.ui)

QT_WRAP_CPP(DIST_MOC_SRCS ${DIST_MOC_HEADERS})

LIST(APPEND SOURCES ${DIST_MOC_SRCS})
LIST(APPEND SOURCES ${DIST_UI_SRCS})
LIST(APPEND HEADERS ${DIST_MOC_HEADERS})


  INCLUDE_DIRECTORIES(
    ${ZLIB_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${PNG_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${OPENSCENEGRAPH_INCLUDE_DIRS}
  )

ADD_COVISE_EXECUTABLE(DistortConfig ${TUI_SOURCES} ${TUI_HEADERS} ${TUI_MOC_HEADERS} ${TUI_MOC_SRCS})
qt_use_modules(DistortConfig Widgets Xml UiTools OpenGL)

TARGET_LINK_LIBRARIES(DistortConfig ${COVISE_NET_LIBRARY} ${COVISE_UTIL_LIBRARY} coOpenPluginUtil coOpenCOVER coOpenVRUI
coOSGVRUI ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_CONFIG_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES} ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(DistortConfig)
