
SET(HEADERS
  
)

SET(SOURCES
  VectorField.cpp
)

SET(EXTRASOURCES
  VectorField.h
)

ADD_COVISE_MODULE(Mapper VectorField ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(VectorField  coAlg coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(VectorField)
