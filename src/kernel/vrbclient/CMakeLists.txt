# @file
# 
# CMakeLists.txt for kernel - vrbclient
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_VRB)

SET(VRBCLIENT_SOURCES
  VRBClient.cpp
  VRBMessage.cpp
)

SET(VRBCLIENT_HEADERS
  VRBClient.h
  VRBMessage.h
)

ADD_COVISE_LIBRARY(coVRBClient ${COVISE_LIB_TYPE} ${VRBCLIENT_SOURCES} ${VRBCLIENT_HEADERS})
TARGET_LINK_LIBRARIES(coVRBClient coNet coConfig)
COVISE_INSTALL_TARGET(coVRBClient)
