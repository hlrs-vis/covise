# @file
# 
# CMakeLists.txt for kernel - appl
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_APPL)

SET(APPL_SOURCES
  CoviseBase.cpp
  ApplInterface.cpp
  RenderInterface.cpp
  CoviseAppModule.cpp
)

SET(APPL_HEADERS
  CoviseBase.h
  ApplInterface.h
  RenderInterface.h
  CoviseAppModule.h
)

ADD_COVISE_LIBRARY(coAppl ${COVISE_LIB_TYPE} ${APPL_SOURCES} ${APPL_HEADERS})
TARGET_LINK_LIBRARIES(coAppl coDo coCore coNet coConfig)

COVISE_WNOERROR(coAppl BASEARCH intrepid)

COVISE_INSTALL_TARGET(coAppl)
COVISE_INSTALL_HEADERS(appl ${APPL_HEADERS})
