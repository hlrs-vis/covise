
SET(HEADERS
  MeanValues.h
)

SET(SOURCES
  MeanValues.cpp
)

SET(EXTRASOURCES
  MeanValues.h
)

ADD_COVISE_MODULE(Tools MeanValues ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(MeanValues  coApi coAppl coCore )

COVISE_INSTALL_TARGET(MeanValues)
