# @file
# 
# CMakeLists.txt for kernel - vrbserver
#
# @author Dennis Grieger

USING(BOOST)

SET(VRBSERVER_SOURCES
  VrbClientList.cpp
  VrbConnectionMap.cpp
  VrbFileBrowserHandler.cpp
  VrbMessageHandler.cpp
  VrbProxie.cpp
  VrbServerRegistry.cpp
  VrbSessionList.cpp
)

SET(VRBSERVER_HEADERS
  VrbClientList.h
  VrbConnectionMap.h
  VrbFileBrowserHandler.h
  VrbMessageHandler.h
  VrbProxie.h
  VrbServerRegistry.h
  VrbSessionList.h
)

ADD_COVISE_LIBRARY(coVRBServer ${COVISE_LIB_TYPE} ${VRBSERVER_SOURCES} ${VRBSERVER_HEADERS})
TARGET_LINK_LIBRARIES(coVRBServer coVRB coNet coUtil coQtUtil coMessages)
COVISE_INSTALL_TARGET(coVRBServer) 
COVISE_INSTALL_HEADERS(vrb/server ${VRBSERVER_HEADERS})
