# @file
# 
# CMakeLists.txt for sys AccessGridDaemon
#
# @author Blasius Czink

SET(ACCESSGRIDDAEMON_SOURCES
  AccessGridDaemon.cpp
)

SET(ACCESSGRIDDAEMON_HEADERS
  AccessGridDaemon.h
)

# EXTRASOURCES    = SharedCovise.py

ADD_COVISE_EXECUTABLE(AccessGridDaemon ${ACCESSGRIDDAEMON_SOURCES} ${ACCESSGRIDDAEMON_HEADERS})
TARGET_LINK_LIBRARIES(AccessGridDaemon coCore coUtil) 

COVISE_INSTALL_TARGET(AccessGridDaemon)
