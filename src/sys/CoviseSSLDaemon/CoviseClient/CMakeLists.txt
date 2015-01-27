# @file
# 
# CMakeLists.txt for sys CoviseSSLDaemon CoviseClient
#
# @author Blasius Czink

INCLUDE_DIRECTORIES(
  ${OPENSSL_INCLUDE_DIR}
)

SET(COVISECLIENT_SOURCES
  CoviseClient.cpp
  SSLClient.cpp
)

SET(COVISECLIENT_HEADERS
  SSLClient.h
)

ADD_COVISE_EXECUTABLE(CoviseClient ${COVISECLIENT_SOURCES} ${COVISECLIENT_HEADERS})
TARGET_LINK_LIBRARIES(CoviseClient coCore coNet coConfig ${OPENSSL_LIBRARIES})

COVISE_INSTALL_TARGET(CoviseClient)
