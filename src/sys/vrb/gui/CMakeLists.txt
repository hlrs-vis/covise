# @file
# 
# CMakeLists.txt for sys vrb vrb (gui)
#
# @author Blasius Czink

ADD_DEFINITIONS(-DGUI)

INCLUDE_DIRECTORIES(
  "${COVISEDIR}/src/OpenCOVER/cover"
  ..
)

SET(VRB_SOURCES
  VRB.cpp
  VRBapplication.cpp
  VRBPopupMenu.cpp
  VRBCurve.cpp
  VRBFileDialog.cpp
  coRegister.cpp
  ../coRegistry.cpp
  ../VRBServer.cpp
  ../VRBClientList.cpp
  "${COVISEDIR}/src/OpenCOVER/cover/coTUIFileBrowser/NetHelp.cpp"
  "${COVISEDIR}/src/OpenCOVER/cover/coTUIFileBrowser/FileSysAccess.cpp"
)

SET(VRB_HEADERS
  ../coRegistry.h
  ../VRBClientList.h
  "${COVISEDIR}/src/OpenCOVER/cover/coTUIFileBrowser/NetHelp.h"
  "${COVISEDIR}/src/OpenCOVER/cover/coTUIFileBrowser/FileSysAccess.h"
)

SET(VRB_MOC_HEADERS
  VRBapplication.h
  VRBPopupMenu.h
  VRBCurve.h
  VRBFileDialog.h
  coRegister.h
  ../VRBServer.h
)

QT_WRAP_CPP(VRB_MOC_SRCS ${VRB_MOC_HEADERS})

ADD_COVISE_EXECUTABLE(vrb ${VRB_SOURCES} ${VRB_HEADERS} ${VRB_MOC_HEADERS} ${VRB_MOC_SRCS})
qt_use_modules(vrb Widgets Network)
IF(BASEARCHSUFFIX STREQUAL "gecko" OR BASEARCHSUFFIX STREQUAL "heiner")
  ADD_COVISE_COMPILE_FLAGS(vrb "-Wno-non-virtual-dtor")
ENDIF()
# COVISE_WNOERROR(coVRML BASEARCH gecko heiner)

TARGET_LINK_LIBRARIES(vrb coVRBClient coNet coUtil)

COVISE_INSTALL_TARGET(vrb)
