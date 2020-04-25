SET(HEADERS
 Profiling.h
 pch.h
 Helper.h
 openGA.h
 SensorPlacement.h
 Zone.h
)
SET(SOURCES
  SensorPlacement.cpp
  Zone.cpp
)
cover_add_plugin(SensorPlacement ${HEADERS} ${SOURCES})