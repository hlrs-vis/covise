SET(HEADERS
)
SET(SOURCES
  MMPTrace.cpp
  SMPTrace.cpp
  TetraGrid.cpp
  TetraTrace.cpp
  WorkstationTrace.cpp
  output.cpp
  trace.cpp
  util.cpp
)
ADD_COVISE_MODULE(Obsolete TetraTrace ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(TetraTrace coApi coAppl coCore)
COVISE_INSTALL_TARGET(TetraTrace)
