SET(HEADERS
 Profiling.h
 SensorPlacement.h
)
SET(SOURCES
  SensorPlacement.cpp
)
cover_add_plugin(SensorPlacement ${HEADERS} ${SOURCES})