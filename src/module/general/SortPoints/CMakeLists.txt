SET(HEADERS
  
)

SET(SOURCES
  SortPoints.cpp
)

SET(EXTRASOURCES
  SortPoints.h
)

ADD_COVISE_MODULE(Filter SortPoints ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(SortPoints  coApi coAppl coCore)

COVISE_INSTALL_TARGET(SortPoints)
