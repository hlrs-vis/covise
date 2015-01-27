
SET(HEADERS
  ComputeTrace.h
)

SET(SOURCES
  ComputeTrace.cpp
)

SET(EXTRASOURCES
  ComputeTrace.h
)

ADD_COVISE_MODULE(Filter  ComputeTrace ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ComputeTrace  coAppl coApi coCore)

COVISE_INSTALL_TARGET(ComputeTrace)
