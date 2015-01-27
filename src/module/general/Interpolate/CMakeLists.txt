
SET(HEADERS
  
)

SET(SOURCES
  Interpolate.cpp
)

SET(EXTRASOURCES
  Interpolate.h
)

ADD_COVISE_MODULE(Interpolator Interpolate ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Interpolate  coApi coAppl coCore )

COVISE_INSTALL_TARGET(Interpolate)
