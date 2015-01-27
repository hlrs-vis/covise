
SET(HEADERS
  
)

SET(SOURCES
  EliminatePolygons.cpp
)

SET(EXTRASOURCES
  EliminatePolygons.h
)

ADD_COVISE_MODULE(Filter EliminatePolygons ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(EliminatePolygons  coApi coAppl coCore )

COVISE_INSTALL_TARGET(EliminatePolygons)
