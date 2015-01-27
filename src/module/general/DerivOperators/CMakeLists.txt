
SET(HEADERS
  
)

SET(SOURCES
  DerivOperators.cpp
)

SET(EXTRASOURCES
  DerivOperators.h
)

ADD_COVISE_MODULE(Tools DerivOperators ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(DerivOperators  coApi coAppl coCore )

COVISE_INSTALL_TARGET(DerivOperators)
