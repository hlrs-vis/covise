
SET(HEADERS
  
)

SET(SOURCES
  LineReduce.cpp
)

SET(EXTRASOURCES
  LineReduce.h
)

ADD_COVISE_MODULE(Filter LineReduce ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(LineReduce  coApi coAppl coCore )

COVISE_INSTALL_TARGET(LineReduce)
