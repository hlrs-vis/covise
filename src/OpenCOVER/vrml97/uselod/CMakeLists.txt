# @file
# 
# CMakeLists.txt for kernel - vrml - uselod
#
# @author Blasius Czink

SET(USELOD_SOURCES
  uselod.cpp
)

ADD_COVISE_EXECUTABLE(uselod ${USELOD_SOURCES})
# TARGET_LINK_LIBRARIES(uselod __) 

COVISE_INSTALL_TARGET(uselod)
