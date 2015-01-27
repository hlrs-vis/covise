
SET(HEADERS
  
)

SET(SOURCES
  AssembleUsg.cpp
)

SET(EXTRASOURCES
  AssembleUsg.h
)

ADD_COVISE_MODULE(Converter AssembleUsg ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(AssembleUsg  coApi coAppl coCore )

COVISE_INSTALL_TARGET(AssembleUsg)
