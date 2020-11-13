# @file
# 
# CMakeLists.txt for kernel - vrbserver
#
# @author Dennis Grieger

USING(BOOST)

SET(VRBSERVER_SOURCES
  VrbClientList.cpp
  VrbFileBrowserHandler.cpp
  VrbMessageHandler.cpp
  VrbServerRegistry.cpp
  VrbSessionList.cpp
)

SET(VRBSERVER_HEADERS
  VrbClientList.h
  VrbFileBrowserHandler.h
  VrbMessageHandler.h
  VrbServerRegistry.h
  VrbSessionList.h
)

ADD_COVISE_LIBRARY(coVRBServer ${COVISE_LIB_TYPE} ${VRBSERVER_SOURCES} ${VRBSERVER_HEADERS})
TARGET_LINK_LIBRARIES(coVRBServer coVRB coNet coUtil coQtUtil)
COVISE_INSTALL_TARGET(coVRBServer) 
COVISE_INSTALL_HEADERS(vrb/server ${VRBSERVER_HEADERS})
