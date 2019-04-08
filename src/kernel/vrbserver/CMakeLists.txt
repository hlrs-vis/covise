# @file
# 
# CMakeLists.txt for kernel - vrbserver
#
# @author Dennis Grieger

USING(BOOST)
ADD_DEFINITIONS(-DGUI)

SET(VRBSERVER_SOURCES
  VrbClientList.cpp
  VrbMessageHandler.cpp
  VrbServerRegistry.cpp
)

SET(VRBSERVER_HEADERS
  VrbClientList.h
  VrbMessageHandler.h
  VrbServerRegistry.h
)

ADD_COVISE_LIBRARY(coVRBServer ${COVISE_LIB_TYPE} ${VRBSERVER_SOURCES} ${VRBSERVER_HEADERS})
TARGET_LINK_LIBRARIES(coVRBServer coVRBClient coNet coUtil coQtUtil)
COVISE_INSTALL_TARGET(coVRBServer) 
COVISE_INSTALL_HEADERS(vrbserver ${VRBSERVER_HEADERS})
