
SET(HEADERS
  
)

SET(SOURCES
  CoverDocument.cpp
)

SET(EXTRASOURCES
  CoverDocument.h
)

ADD_COVISE_MODULE(IO CoverDocument ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(CoverDocument  coApi coAppl coCore )

COVISE_INSTALL_TARGET(CoverDocument)
