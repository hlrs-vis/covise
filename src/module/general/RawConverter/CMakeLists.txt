
SET(HEADERS
  
)

SET(SOURCES
    RawConverter.cpp
)

SET(EXTRASOURCES
    RawConverter.h
)

ADD_COVISE_MODULE(Tools RawConverter ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(RawConverter  coApi coAppl coCore )

COVISE_INSTALL_TARGET(RawConverter)
