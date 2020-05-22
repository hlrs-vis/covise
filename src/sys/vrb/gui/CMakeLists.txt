# @file
# 
# CMakeLists.txt for sys vrb vrb (gui)
#
# @author Blasius Czink
USING(BOOST)
ADD_DEFINITIONS(-DGUI)

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

SET(VRB_HEADERS
  ../coRegistry.h
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

QT_WRAP_CPP(VRB_MOC_SRCS ${VRB_MOC_HEADERS})

ADD_COVISE_EXECUTABLE(vrb ${VRB_SOURCES} ${VRB_HEADERS} ${VRB_MOC_HEADERS} ${VRB_MOC_SRCS})
qt_use_modules(vrb Widgets Network)
IF(BASEARCHSUFFIX STREQUAL "gecko" OR BASEARCHSUFFIX STREQUAL "heiner")
  ADD_COVISE_COMPILE_FLAGS(vrb "-Wno-non-virtual-dtor")
ENDIF()
# COVISE_WNOERROR(coVRML BASEARCH gecko heiner)

TARGET_LINK_LIBRARIES(vrb coVRBServer coVRBClient coNet coUtil coQtUtil)

COVISE_INSTALL_TARGET(vrb)
