
SET(HEADERS
  
)

SET(SOURCES
  SplitGeometry.cpp
)

SET(EXTRASOURCES
  SplitGeometry.h
)

ADD_COVISE_MODULE(Tools SplitGeometry ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(SplitGeometry  coApi coAppl coCore)

COVISE_INSTALL_TARGET(SplitGeometry)
