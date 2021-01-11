# @file
# 
# CMakeLists.txt for sys controller
#
# @author Blasius Czink

IF(COVISE_BUILD_WEBSERVICE)
  ADD_DEFINITIONS(-DHAVE_GSOAP)
ENDIF()

SET(CONTROLLER_SOURCES
  config.cpp
  exception.cpp
  util.cpp
  netLink.cpp
  object.cpp
  port.cpp
  subProcess.cpp
  userinterface.cpp
  controlProcess.cpp
  module.cpp
  crb.cpp
  host.cpp
  renderModule.cpp
  modui.cpp
  global.cpp
  handler.cpp
  main.cpp
  moduleInfo.cpp)

SET(CONTROLLER_HEADERS
  config.h
  controlProcess.h
  crb.h
  exception.h
  global.h
  handler.h
  host.h
  list.h
  modui.h
  module.h
  moduleInfo.h
  netLink.h
  object.h
  port.h
  subProcess.h
  renderModule.h
  userinterface.h
  util.h
)

ADD_COVISE_EXECUTABLE(covise ${CONTROLLER_SOURCES} ${CONTROLLER_HEADERS})
TARGET_LINK_LIBRARIES(covise coAppl coApi coCore coConfig coUtil coVRBClient coMessages) 
qt_use_modules(covise Core)

COVISE_INSTALL_TARGET(covise)
