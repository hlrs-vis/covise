SET(HEADERS
  
)

SET(SOURCES
  WritePatran.cpp
)

SET(EXTRASOURCES
  WritePatran.h
)

ADD_COVISE_MODULE(IO WritePatran ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(WritePatran  coApi coAppl coCore )

COVISE_INSTALL_TARGET(WritePatran)
