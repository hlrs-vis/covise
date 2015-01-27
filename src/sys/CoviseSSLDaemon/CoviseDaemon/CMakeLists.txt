# @file
# 
# CMakeLists.txt for sys CoviseSSLDaemon CoviseDaemon (CoviseDaemon)
#
# @author Blasius Czink

# setup what we need from Qt
INCLUDE_DIRECTORIES(
  ${OPENSSL_INCLUDE_DIR}
  ../../vrb
)

SET(COVISEDAEMON_SOURCES
  CoviseDaemon.cpp
  frmAbout.cpp
  frmMainWindow.cpp
  frmRequestDialog.cpp
  SSLDaemon.cpp
  string_utils.cpp
  ../../vrb/coRegistry.cpp
  ../../vrb/VRBClientList.cpp
)

SET(COVISEDAEMON_HEADERS
  string_utils.h 
)

SET(COVISEDAEMON_MOC_HEADERS
  frmAbout.h
  frmMainWindow.h
  frmRequestDialog.h
  SSLDaemon.h
)

QT_WRAP_CPP(COVISEDAEMON_MOC_SRCS ${COVISEDAEMON_MOC_HEADERS})

ADD_COVISE_EXECUTABLE(CoviseDaemon ${COVISEDAEMON_SOURCES} ${COVISEDAEMON_HEADERS} ${COVISEDAEMON_MOC_SRCS} ${COVISEDAEMON_MOC_HEADERS})
qt_use_modules(CoviseDaemon Widgets Xml)
TARGET_LINK_LIBRARIES(CoviseDaemon coCore coNet coConfig ${OPENSSL_LIBRARIES})

COVISE_INSTALL_TARGET(CoviseDaemon)

IF(APPLE)
   COVISE_WNOERROR(CoviseDaemon)
ENDIF(APPLE)
