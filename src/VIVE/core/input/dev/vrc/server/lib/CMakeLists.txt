# @file
# 
# CMakeLists.txt for sys DevServer lib (coDevServer)
#
# @author Blasius Czink

ADD_DEFINITIONS(-DSTANDALONE)

IF(WIN32)
  ADD_DEFINITIONS(-DUTIL_EXPORTS -DCOVISE_UTIL)
ENDIF(WIN32)

SET(UTIL ${COVISEDIR}/src/kernel/util)

SET(CODEVSERVER_SOURCES
   ${UTIL}/ArgsParser.cpp
   ${UTIL}/SerialCom.cpp
   ${UTIL}/UDP_Sender.cpp
)

SET(CODEVSERVER_HEADERS
   ${UTIL}/ArgsParser.h
   ${UTIL}/SerialCom.h
   ${UTIL}/UDP_Sender.h
)

ADD_COVISE_LIBRARY(coDevServer STATIC ${CODEVSERVER_SOURCES} ${CODEVSERVER_HEADERS})
# TARGET_LINK_LIBRARIES(coDevServer ___)

COVISE_INSTALL_TARGET(coDevServer)
