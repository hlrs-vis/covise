# @file
# 
# CMakeLists.txt for sys vrb console
#
# @author Blasius Czink

INCLUDE_DIRECTORIES(
  "${COVISEDIR}/src/OpenCOVER/cover"
  ..
)

SET(VRBC_SOURCES
  VRBC.cpp
  ../VRBServer.cpp
  ../coRegistry.cpp
  ../VRBClientList.cpp
  "${COVISEDIR}/src/OpenCOVER/cover/coTUIFileBrowser/NetHelp.cpp"
  "${COVISEDIR}/src/OpenCOVER/cover/coTUIFileBrowser/FileSysAccess.cpp"
)

SET(VRBC_HEADERS
  ../VRBServer.h
  ../coRegistry.h
  ../VRBClientList.h
  "${COVISEDIR}/src/OpenCOVER/cover/coTUIFileBrowser/NetHelp.h"
  "${COVISEDIR}/src/OpenCOVER/cover/coTUIFileBrowser/FileSysAccess.h"
)

ADD_COVISE_EXECUTABLE(vrbc ${VRBC_SOURCES} ${VRBC_HEADERS})
TARGET_LINK_LIBRARIES(vrbc coVRBClient coCore coNet coUtil) 
qt_use_modules(vrbc Core Network)

COVISE_INSTALL_TARGET(vrbc)
