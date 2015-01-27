
SET(HEADERS
  
)

SET(SOURCES
  Sample.cpp
  unstruct.cpp
)

SET(EXTRASOURCES
  Sample.h
  unstruct.h
)

ADD_COVISE_MODULE(Interpolator Sample ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Sample  coApi coAppl coCore )

COVISE_INSTALL_TARGET(Sample)
