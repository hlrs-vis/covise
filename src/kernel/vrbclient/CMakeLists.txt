# @file
# 
# CMakeLists.txt for kernel - vrbclient
#
# @author Blasius Czink

USING(BOOST)
USING(Pthreads)
ADD_DEFINITIONS(-DCOVISE_VRB)

SET(VRBCLIENT_SOURCES
  RegistryClass.cpp
  RegistryVariable.cpp
  SessionID.cpp	
  SharedState.cpp
  SharedStateManager.cpp
  SharedStateSerializer.cpp
  VRBClient.cpp
  VRBMessage.cpp
  VrbClientRegistry.cpp
  VrbRegistry.cpp
)

SET(VRBCLIENT_HEADERS
  RegistryClass.h
  RegistryVariable.h
  SessionID.h
  SharedMap.h
  SharedState.h
  SharedStateManager.h
  SharedStateSerializer.h
  VRBClient.h
  VRBMessage.h
  VrbClientRegistry.h
  VrbMessageSenderInterface.h
  VrbRegistry.h
)

ADD_COVISE_LIBRARY(coVRBClient ${COVISE_LIB_TYPE} ${VRBCLIENT_SOURCES} ${VRBCLIENT_HEADERS})
TARGET_LINK_LIBRARIES(coVRBClient coNet coConfig)
COVISE_INSTALL_TARGET(coVRBClient)
COVISE_INSTALL_HEADERS(vrbclient ${VRBCLIENT_HEADERS})
