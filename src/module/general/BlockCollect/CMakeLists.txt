
SET(HEADERS
  
)

SET(SOURCES
  BlockCollect.cpp
)

SET(EXTRASOURCES
  BlockCollect.h
)

ADD_COVISE_MODULE(Tools BlockCollect ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(BlockCollect  coApi coAppl coCore )

COVISE_INSTALL_TARGET(BlockCollect)
