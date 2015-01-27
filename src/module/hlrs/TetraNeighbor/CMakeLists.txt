SET(HEADERS
)
SET(SOURCES
  SMP.cpp
  TetraNeighbor.cpp
)
ADD_COVISE_MODULE(Obsolete TetraNeighbor ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(TetraNeighbor coApi coAppl coCore)
COVISE_INSTALL_TARGET(TetraNeighbor)
