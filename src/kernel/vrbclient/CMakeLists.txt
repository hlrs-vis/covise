# @file
# 
# CMakeLists.txt for kernel - vrbclient
#
# @author Blasius Czink

USING(BOOST)
ADD_DEFINITIONS(-DCOVISE_VRB)

SET(VRBCLIENT_SOURCES
  VRBClient.cpp
  VRBMessage.cpp
  VrbRegistry.cpp
  VrbClientRegistry.cpp
  regClass.cpp
  SharedState.cpp
  SharedStateManager.cpp
  ShareStateSerializer.cpp
)

SET(VRBCLIENT_HEADERS
  VRBClient.h
  VRBMessage.h
  VrbRegistry.h
  VrbClientRegistry.h
  regClass.h
  SharedState.h
  SharedStateManager.h
  ShareStateSerializer.h
)

ADD_COVISE_LIBRARY(coVRBClient ${COVISE_LIB_TYPE} ${VRBCLIENT_SOURCES} ${VRBCLIENT_HEADERS})
TARGET_LINK_LIBRARIES(coVRBClient coNet coConfig)
COVISE_INSTALL_TARGET(coVRBClient)
COVISE_INSTALL_HEADERS(vrbclient ${VRBCLIENT_HEADERS})
