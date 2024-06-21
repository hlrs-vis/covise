USING(CURL)

SET(SOURCES
  request.cpp
  methods.cpp
)

SET(HEADERS
  export.h
  methods.h
  request.h
)

ADD_COVISE_LIBRARY(coCurlHTTPClient ${SOURCES} ${HEADERS})
COVISE_INSTALL_TARGET(coCurlHTTPClient)

COVISE_WNOERROR(coCurlHTTPClient BASEARCH zebu)