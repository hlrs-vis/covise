
SET(HEADERS
  
)

SET(SOURCES
  TensorScal.cpp
  jacobi_eigenvalue.cpp
)

SET(EXTRASOURCES
  TensorScal.h
  jacobi_eigenvalue.hpp
)

ADD_COVISE_MODULE(Tools TensorScal ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(TensorScal  coApi coAppl coCore )

COVISE_INSTALL_TARGET(TensorScal)
