
SET(HEADERS
  
)

SET(SOURCES
  StretchSet.cpp
)

SET(EXTRASOURCES
  StretchSet.h
)

ADD_COVISE_MODULE(Filter StretchSet ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(StretchSet  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(StretchSet)
