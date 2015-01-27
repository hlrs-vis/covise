
SET(HEADERS
  
)

SET(SOURCES
  PerformerScene.cpp
)

SET(EXTRASOURCES
  PerformerScene.h
)

ADD_COVISE_MODULE(IO PerformerScene ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(PerformerScene  coApi coAppl coCore )

COVISE_INSTALL_TARGET(PerformerScene)
