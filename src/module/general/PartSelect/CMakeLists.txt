
SET(HEADERS
  
)

SET(SOURCES
  PartSelect.cpp
)

SET(EXTRASOURCES
  PartSelect.h
)

ADD_COVISE_MODULE(Filter PartSelect ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(PartSelect  coApi coAppl coCore )

COVISE_INSTALL_TARGET(PartSelect)
