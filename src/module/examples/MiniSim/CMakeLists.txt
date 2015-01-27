SET(HEADERS
  MiniSim.h
)
SET(SOURCES
  MiniSim.cpp
)
covise_add_module(Examples MiniSim ${EXTRASOURCES} ${SOURCES} ${HEADERS})
