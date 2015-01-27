
SET(HEADERS
  
)

SET(SOURCES
  ColorDistance.cpp
)

SET(EXTRASOURCES
  ColorDistance.h
)

ADD_COVISE_MODULE(Filter ColorDistance ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ColorDistance  coApi coAppl coCore )

COVISE_INSTALL_TARGET(ColorDistance)
