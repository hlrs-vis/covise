# @file
# 
# CMakeLists.txt for application general RWCovise
#
# @author Blasius Czink

SET(SOURCES
  RWCovise.cpp
)

SET(HEADERS
  RWCovise.h
)

ADD_COVISE_MODULE(IO RWCovise)
TARGET_LINK_LIBRARIES(RWCovise coReader coApi coAppl coCore) 

COVISE_INSTALL_TARGET(RWCovise)
