SET(HEADERS
  ReadTrajectory.h
)
SET(SOURCES
ReadTrajectory.cpp
)
covise_add_module(ReadTrajectory ${EXTRASOURCES} ${SOURCES} ${HEADERS})
