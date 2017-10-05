# @file
# 
# CMakeLists.txt for renderer QtRenderer
#
# @author Blasius Czink

PROJECT(QtRenderer CXX C)

USING(Virvo optional)

if (WIN32)
    COVISE_FIND_PACKAGE(Coin3D REQUIRED)
    COVISE_FIND_PACKAGE(SoQt REQUIRED)
else()
    COVISE_FIND_PACKAGE(Coin3D)
    COVISE_FIND_PACKAGE(SoQt)
endif()

IF(NOT COIN3D_FOUND OR NOT SOQT_FOUND)
    RETURN()
ENDIF()


IF(WIN32)
  ADD_DEFINITIONS(-DCOIN_DLL -DSOQT_DLL)
  IF(MSVC)
    SET(EXTRA_LIBS ${EXTRA_LIBS} winmm)
  ENDIF(MSVC)
ENDIF(WIN32)

IF(APPLE)
  # with the Inventor from mac ports, we had
  # problems with a function declaration from
  # a template that could not be resolved.
  # Luckily, that function is only used if
  # NDEBUG is not defined
  ADD_DEFINITIONS(-DNDEBUG)
ENDIF(APPLE)

INCLUDE_DIRECTORIES(
  ${OPENGL_INCLUDE_DIR}
  ${COIN3D_INCLUDE_DIRS}
  ${SOQT_INCLUDE_DIRS}
)

SET(HEADERS
  CoviseWindowCapture.h
  InvCommunicator.h
  InvComposePlane.h
  InvManipList.h
  InvObjectList.h
  InvObjectManager.h
  InvObjects.h
  InvPlaneMover.h
  InvTelePointer.h
  InvViewer.h
  SoBillboard.h
  objTreeWidget.h
  objTreeItem.h
  colormapListWidget.h
  colormapListItem.h
  InvTextManager.h
)

SET(SOURCES
  CoviseWindowCapture.cpp
  InvClipPlaneEditor.cpp
  InvCommunicator.cpp
  InvComposePlane.cpp
  InvMain.cpp
  InvManipList.cpp
  InvMsgManager.cpp
  InvObjectList.cpp
  InvObjectManager.cpp
  InvObjects.cpp
  InvPlaneMover.cpp
  InvSequencer.cpp
  InvTelePointer.cpp
  InvViewer.cpp
  main.cpp
  SoBillboard.cpp
  objTreeWidget.cpp
  objTreeItem.cpp
  colormapListWidget.cpp
  colormapListItem.cpp
  InvTextManager.cpp
)

if (COVISE_USE_VIRVO)
   set(HEADERS ${HEADERS}
      SoVolumeDetail.h
      SoVolume.h
      )
   set(SOURCES ${SOURCES}
      SoVolume.cpp
      SoVolumeDetail.cpp
      )
endif()

SET(QtRenderer_MOCS
  InvClipPlaneEditor.h
  InvMain.h
  InvMsgManager.h
  InvSequencer.h
  objTreeWidget.h
  objTreeItem.h
  colormapListWidget.h
  colormapListItem.h
)
QT_WRAP_CPP(MOCS ${QtRenderer_MOCS})

ADD_COVISE_MODULE(Renderer QtRenderer MACOSX_BUNDLE ${QtRenderer_MOCS} ${MOCS})
# setup what we need from Qt
qt_use_modules(QtRenderer Xml Widgets)


TARGET_LINK_LIBRARIES(QtRenderer
  coAppl coCore coUtil
  ${COIN3D_LIBRARIES} ${SOQT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${EXTRA_LIBS}
)

COVISE_INSTALL_TARGET(QtRenderer)
