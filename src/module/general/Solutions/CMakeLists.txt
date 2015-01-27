
SET(HEADERS
  
)

SET(SOURCES
  Solutions.cpp
)

SET(EXTRASOURCES
  Solutions.h
)

ADD_COVISE_MODULE(Tools Solutions ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Solutions  coApi coAppl coCore )

COVISE_INSTALL_TARGET(Solutions)
