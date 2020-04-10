SET(HEADERS
 Profiling.h
 openGA.h
 SensorPlacement.h
)
SET(SOURCES
  SensorPlacement.cpp
)
cover_add_plugin(SensorPlacement ${HEADERS} ${SOURCES})