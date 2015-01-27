
SET(HEADERS
  
)

SET(SOURCES
  Scalar2Vector.cpp
)

SET(EXTRASOURCES
  Scalar2Vector.h
)

ADD_COVISE_MODULE(Converter Scalar2Vector ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Scalar2Vector  coApi coAppl coCore )

COVISE_INSTALL_TARGET(Scalar2Vector)
