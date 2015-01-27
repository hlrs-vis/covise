
SET(HEADERS
  
)

SET(SOURCES
  VectorScal.cpp
)

SET(EXTRASOURCES
  VectorScal.h
)

ADD_COVISE_MODULE(Tools VectorScal ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(VectorScal  coApi coAppl coCore )

COVISE_INSTALL_TARGET(VectorScal)
