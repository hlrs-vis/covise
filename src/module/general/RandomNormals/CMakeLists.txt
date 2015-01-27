SET(HEADERS
  
)

SET(SOURCES
  RandomNormals.cpp
)

SET(EXTRASOURCES
  RandomNormals.h
)

ADD_COVISE_MODULE(Converter RandomNormals ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(RandomNormals  coApi coAppl coCore )

COVISE_INSTALL_TARGET(RandomNormals)
