# @file
# 
# CMakeLists.txt for kernel - star
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_STAR)

SET(STAR_SOURCES
  File09.cpp
  File16.cpp
  File29.cpp
  IllConv.cpp
  istreamBLK.cpp
  istreamFTN.cpp
  SammConv.cpp
  StarFile.cpp
  starInfo.cpp
)

SET(STAR_HEADERS
  File16.h
  IllConv.h
  istreamFTN.h
  StarFile.h
  File09.h
  File29.h
  istreamBLK.h
  SammConv.h
)

ADD_COVISE_LIBRARY(coStar ${COVISE_LIB_TYPE} ${STAR_SOURCES} ${STAR_HEADERS})
TARGET_LINK_LIBRARIES(coStar coCore coApi coAppl)

COVISE_INSTALL_TARGET(coStar)
