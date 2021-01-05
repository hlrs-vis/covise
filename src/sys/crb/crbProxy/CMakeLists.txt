# @file
# 
# CMakeLists.txt for sys crbProxy
#
# @author Blasius Czink

SET(CRBPROXY_SOURCES
  CRBConnection.cpp
  Proxy.cpp
  crbProxy.cpp
)

SET(CRBPROXY_HEADERS
  CRBConnection.h
  Proxy.h
)

ADD_COVISE_EXECUTABLE(crbProxy ${CRBPROXY_SOURCES} ${CRBPROXY_HEADERS})
TARGET_LINK_LIBRARIES(crbProxy coDmgr coCore coUtil coNet coMessages) 

COVISE_INSTALL_TARGET(crbProxy)
