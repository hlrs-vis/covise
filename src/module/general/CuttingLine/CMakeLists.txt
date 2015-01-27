
SET(HEADERS
  
)

SET(SOURCES
  CuttingLine.cpp
)

SET(EXTRASOURCES
  CuttingLine.h
)

ADD_COVISE_MODULE(Filter CuttingLine ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(CuttingLine  coApi coAppl coCore )

COVISE_INSTALL_TARGET(CuttingLine)
