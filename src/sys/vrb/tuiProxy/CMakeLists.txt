# @file
# 
# CMakeLists.txt for sys vrb tuiProxy
#
# @author Blasius Czink

SET(TUIPROXY_SOURCES
  TUIProxy.cpp
)

SET(TUIPROXY_HEADERS
  TUIProxy.h
)

ADD_COVISE_EXECUTABLE(tuiProxy ${TUIPROXY_SOURCES} ${TUIPROXY_HEADERS})
TARGET_LINK_LIBRARIES(tuiProxy coNet coUtil) 

COVISE_INSTALL_TARGET(tuiProxy)
