
SET(HEADERS
  
)

SET(SOURCES
  ReduceSet.cpp
)

SET(EXTRASOURCES
  ReduceSet.h
)

ADD_COVISE_MODULE(Filter ReduceSet ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReduceSet  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(ReduceSet)
