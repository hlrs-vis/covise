
SET(HEADERS
  
)

SET(SOURCES
  OceanDepth.cpp
)

SET(EXTRASOURCES
  OceanDepth.h
)

ADD_COVISE_MODULE(Tools OceanDepth ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(OceanDepth coApi coAppl coCore )

COVISE_INSTALL_TARGET(OceanDepth)
