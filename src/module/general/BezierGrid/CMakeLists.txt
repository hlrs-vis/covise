
SET(HEADERS
  BezierGrid.h
)

SET(SOURCES
  BezierGrid.cpp
)

SET(EXTRASOURCES
  BezierGrid.h
)

ADD_COVISE_MODULE(Filter  BezierGrid ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES (BezierGrid  coAppl coApi coCore)

COVISE_INSTALL_TARGET(BezierGrid)
