
SET(HEADERS
  
)

SET(SOURCES
  VariantMarker.cpp
)

SET(EXTRASOURCES
  VariantMarker.h
)

ADD_COVISE_MODULE(Filter VariantMarker ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(VariantMarker  coApi coAppl coCore )

COVISE_INSTALL_TARGET(VariantMarker)
