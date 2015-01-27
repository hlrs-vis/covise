
SET(HEADERS
  
)

SET(SOURCES
  GenNormals.cpp
)

SET(EXTRASOURCES
  GenNormals.h
)

ADD_COVISE_MODULE(Tools GenNormals ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(GenNormals  coApi coAppl coCore )

COVISE_INSTALL_TARGET(GenNormals)
