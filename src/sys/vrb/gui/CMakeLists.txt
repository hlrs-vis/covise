# @file
# 
# CMakeLists.txt for sys vrb vrb (gui)
#
# @author Blasius Czink

COVISE_FIND_PACKAGE(Qt5 REQUIRED 
COMPONENTS 
Core
Gui
Widgets
)

USING(BOOST)

ADD_DEFINITIONS(-DGUI)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

INCLUDE_DIRECTORIES(
  ..
)

SET(VRB_SOURCES
  VRB.cpp
  VRBapplication.cpp
  VRBPopupMenu.cpp
  VRBCurve.cpp
  VRBFileDialog.cpp
  coRegister.cpp
  ../VRBServer.cpp
  ../VrbUiClientList.cpp
  ../VrbUiMessageHandler.cpp
)

SET(VRB_MOC_HEADERS
  VRBapplication.h
  VRBPopupMenu.h
  VRBCurve.h
  VRBFileDialog.h
  coRegister.h
  ../VRBServer.h
  ../VrbUiClientList.h
  ../VrbUiMessageHandler.h
)

SET(VRB_MOC_SRCS
  resources.qrc
)

ADD_COVISE_EXECUTABLE(vrb ${VRB_SOURCES} ${VRB_MOC_HEADERS} ${VRB_MOC_SRCS})
IF(BASEARCHSUFFIX STREQUAL "gecko" OR BASEARCHSUFFIX STREQUAL "heiner")
  ADD_COVISE_COMPILE_FLAGS(vrb "-Wno-non-virtual-dtor")
ENDIF()
# COVISE_WNOERROR(coVRML BASEARCH gecko heiner)

TARGET_LINK_LIBRARIES(vrb
                        Qt5::Core
                        Qt5::Gui
                        Qt5::Widgets
                        coNet
                        coQtUtil
                        coUtil
                        coVRB
                        coVRBServer)

COVISE_INSTALL_TARGET(vrb)
