# @file
# 
# CMakeLists.txt for application general RWCovise
#
# @author Blasius Czink

FIND_PACKAGE(CFX)
IF (NOT CFX_FOUND)
   RETURN()
ENDIF(NOT CFX_FOUND)

INCLUDE_DIRECTORIES(${CFX_INCLUDE_DIRS})

SET(SOURCES
  ReadCFX.cpp
  ../Transform/Matrix.cpp
)

SET(HEADERS
  ReadCFX.h
)

ADD_COVISE_MODULE(IO ReadCFX)
TARGET_LINK_LIBRARIES(ReadCFX coReader coApi coAppl coCore ${CFX_LIBRARIES}) 

COVISE_INSTALL_TARGET(ReadCFX)
