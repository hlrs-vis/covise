# @file
# 
# CMakeLists.txt for sys controller
#
# @author Blasius Czink

USING(BOOST)

IF(COVISE_BUILD_WEBSERVICE)
  ADD_DEFINITIONS(-DHAVE_GSOAP)
ENDIF()

SET(CONTROLLER_SOURCES
  config.cpp
  controlProcess.cpp
  crb.cpp
  exception.cpp
  global.cpp
  handler.cpp
  host.cpp
  hostManager.cpp
  main.cpp
  modui.cpp
  module.cpp
  moduleInfo.cpp
  netLink.cpp
  object.cpp
  port.cpp
  proxyConnection.cpp
  renderModule.cpp
  subProcess.cpp
  userinterface.cpp
  util.cpp
  )

SET(CONTROLLER_HEADERS
  config.h
  controlProcess.h
  crb.h
  exception.h
  global.h
  handler.h
  host.h
  hostManager.h
  list.h
  modui.h
  module.h
  moduleInfo.h
  netLink.h
  object.h
  port.h
  proxyConnection.h
  renderModule.h
  subProcess.h
  syncVar.h
  userinterface.h
  util.h
)

ADD_COVISE_EXECUTABLE(covise ${CONTROLLER_SOURCES} ${CONTROLLER_HEADERS})
TARGET_LINK_LIBRARIES(covise coAppl coApi coCore coConfig coUtil coVRBClient coMessages) 
qt_use_modules(covise Core)

COVISE_INSTALL_TARGET(covise)
