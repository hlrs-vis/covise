
SET(HEADERS
  
)

SET(SOURCES
  IndexManifolds.cpp
)

SET(EXTRASOURCES
  IndexManifolds.h
)

ADD_COVISE_MODULE(Filter IndexManifolds ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(IndexManifolds  coApi coAppl coCore )

COVISE_INSTALL_TARGET(IndexManifolds)
