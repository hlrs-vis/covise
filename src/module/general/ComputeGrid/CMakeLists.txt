
SET(HEADERS
  ComputeGrid.h
)

SET(SOURCES
  ComputeGrid.cpp
)

SET(EXTRASOURCES
  ComputeGrid.h
)

ADD_COVISE_MODULE(Filter  ComputeGrid ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ComputeGrid  coAppl coApi coCore)

COVISE_INSTALL_TARGET(ComputeGrid)
