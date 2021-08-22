
ADD_DEFINITIONS(-DCOVISE_RS)

SET(RSCLIENT_SOURCES
  remoteSoundClient.cpp
  remoteSound.cpp
)

SET(RSCLIENT_HEADERS
  remoteSoundClient.h
  remoteSound.h
  ../../sys/carSound/remoteSoundMessages.h
)

ADD_COVISE_LIBRARY(coRSClient ${COVISE_LIB_TYPE} ${RSCLIENT_SOURCES} ${RSCLIENT_HEADERS})
TARGET_LINK_LIBRARIES(coRSClient coNet coConfig)
COVISE_INSTALL_TARGET(coRSClient)
COVISE_INSTALL_HEADERS(rs/client ${RSCLIENT_HEADERS})
