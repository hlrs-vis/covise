# @file
# 
# CMakeLists.txt for sys vrb vrbProxy
#
# @author Blasius Czink

SET(VRBPROXY_SOURCES
  VRBProxy.cpp
)

SET(VRBPROXY_HEADERS
  VRBProxy.h
)

ADD_COVISE_EXECUTABLE(vrbProxy ${VRBPROXY_SOURCES} ${VRBPROXY_HEADERS})
TARGET_LINK_LIBRARIES(vrbProxy coNet coCore coUtil) 

COVISE_INSTALL_TARGET(vrbProxy)
