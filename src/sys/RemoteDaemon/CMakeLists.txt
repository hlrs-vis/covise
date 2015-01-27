# @file
# 
# CMakeLists.txt for sys RemoteDaemon
#
# @author Blasius Czink

SET(REMOTEDAEMON_SOURCES
  RemoteDaemon.cpp
)

SET(REMOTEDAEMON_HEADERS
  RemoteDaemon.h
)

ADD_COVISE_EXECUTABLE(RemoteDaemon ${REMOTEDAEMON_SOURCES} ${REMOTEDAEMON_HEADERS})
TARGET_LINK_LIBRARIES(RemoteDaemon coNet coCore coUtil coConfig) 

COVISE_INSTALL_TARGET(RemoteDaemon)
