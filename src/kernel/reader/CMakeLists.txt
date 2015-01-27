# @file
# 
# CMakeLists.txt for kernel - reader
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_READER)

SET(READER_SOURCES
  coReader.cpp
  CoviseIO.cpp
  Items.cpp
  ReaderControl.cpp
)

SET(READER_HEADERS
  coReader.h
  CoviseIO.h
  Items.h
  ReaderControl.h
)

ADD_COVISE_LIBRARY(coReader ${COVISE_LIB_TYPE} ${READER_SOURCES} ${READER_HEADERS})
TARGET_LINK_LIBRARIES(coReader coCore coApi coAppl coFile)

COVISE_INSTALL_TARGET(coReader)
