# @file
# 
# CMakeLists.txt for kernel - file
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_FILE)

SET(FILE_SOURCES
  covFiles.c
)

SET(FILE_HEADERS
  coFileExport.h
  covFiles.h
  covReadFiles.h
  covWriteFiles.h 
)

ADD_COVISE_LIBRARY(coFile ${COVISE_LIB_TYPE} ${FILE_SOURCES} ${FILE_HEADERS})
COVISE_INSTALL_TARGET(coFile)
COVISE_INSTALL_HEADERS(file ${FILE_HEADERS})
