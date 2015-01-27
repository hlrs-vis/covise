
SET(HEADERS
  
)

SET(SOURCES
  TensorScal.cpp
)

SET(EXTRASOURCES
  TensorScal.h
)

ADD_COVISE_MODULE(Tools TensorScal ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(TensorScal  coApi coAppl coCore )

COVISE_INSTALL_TARGET(TensorScal)
