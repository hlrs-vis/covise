
SET(HEADERS
  
)

SET(SOURCES
  WritePolygon.cpp
)

SET(EXTRASOURCES
  WritePolygon.h
)

ADD_COVISE_MODULE(IO WritePolygon ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(WritePolygon  coApi coAppl coCore )

COVISE_INSTALL_TARGET(WritePolygon)
