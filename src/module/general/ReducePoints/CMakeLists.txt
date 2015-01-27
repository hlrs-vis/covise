SET(HEADERS
  
)

SET(SOURCES
  ReducePoints.cpp
)

SET(EXTRASOURCES
  ReducePoints.h
)

ADD_COVISE_MODULE(Converter ReducePoints ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReducePoints  coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReducePoints)
