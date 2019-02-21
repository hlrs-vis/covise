# @file
# 
# CMakeLists.txt for sys vrb console
#
# @author Blasius Czink

INCLUDE_DIRECTORIES(
  ..
)

SET(VRBC_SOURCES
  VRBC.cpp
  ../VRBServer.cpp
  ../coRegistry.cpp
  ../VRBClientList.cpp
  ../VrbServerRegistry.cpp
)

SET(VRBC_HEADERS
  ../VRBServer.h
  ../coRegistry.h
  ../VRBClientList.h
  ../VrbServerRegistry.h
)

ADD_COVISE_EXECUTABLE(vrbc ${VRBC_SOURCES} ${VRBC_HEADERS})
TARGET_LINK_LIBRARIES(vrbc coVRBClient coNet coUtil coQtUtil)
qt_use_modules(vrbc Core Network)

COVISE_INSTALL_TARGET(vrbc)
