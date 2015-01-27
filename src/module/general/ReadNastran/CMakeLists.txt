SET(HEADERS
  
)

SET(SOURCES
  ReadNastran.cpp
)

SET(EXTRASOURCES
  ReadNastran.h
  elemList.h
)

ADD_COVISE_MODULE(IO ReadNastran ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadNastran  coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadNastran)
