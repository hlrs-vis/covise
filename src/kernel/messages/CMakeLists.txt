set(CO_MSG_HEADER
  CRB_EXEC.h
  NEW_UI.h
  PROXY.h
  VRB_PERMIT_LAUNCH.h
  coviseLaunchOptions.h
)

set(CO_MSG_SOURCES
  CRB_EXEC.cpp
  NEW_UI.cpp
  PROXY.cpp
  VRB_PERMIT_LAUNCH.cpp
)

ADD_COVISE_LIBRARY(coMessages ${CO_MSG_HEADER} ${CO_MSG_SOURCES})
target_link_libraries(coMessages
                      coNet
                      coVRBClient
                      coUtil
)

COVISE_INSTALL_TARGET(coMessages)
COVISE_INSTALL_HEADERS(net ${CO_MSG_HEADER})