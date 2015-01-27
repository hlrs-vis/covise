
SET(HEADERS
  PurgeCoords.h
)

SET(SOURCES
  PurgeCoords.cpp
)

SET(EXTRASOURCES
  PurgeCoords.h
)

ADD_COVISE_MODULE(Filter PurgeCoords )
TARGET_LINK_LIBRARIES(PurgeCoords  coApi coAppl coCore )

COVISE_INSTALL_TARGET(PurgeCoords)
