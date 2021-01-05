# @file
# 
# CMakeLists.txt for sys crb
#
# @author Blasius Czink

IF(UNIX)
  # TODO: add FLEXLM support on unix
  
ENDIF(UNIX)

SET(CRB_SOURCES
  CRB_Module.cpp
  crb.cpp
)

SET(CRB_HEADERS
  CRB_Module.h
)

ADD_COVISE_EXECUTABLE(crb ${CRB_SOURCES} ${CRB_HEADERS})
TARGET_LINK_LIBRARIES(crb coApi coAppl coVRBClient coDmgr coCore coUtil coNet coMessages) 
qt_use_modules(crb Core Xml)

COVISE_INSTALL_TARGET(crb)
