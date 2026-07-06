SET(HEADERS
 Profiling.h
 openGA.h
 SensorPlacement.h
 Zone.h
 Sensor.h
 UI.h
 DataManager.h
 Factory.h
 GA.h
 Camera.h
 UDP.h
 GridPoint.h
)

SET(SOURCES
  SensorPlacement.cpp
  Zone.cpp
  Sensor.cpp
  UI.cpp
  DataManager.cpp
  Factory.cpp
  GA.cpp
  Camera.cpp
  UDP.cpp
  GridPoint.cpp
)


cover_add_plugin(SensorPlacement ${HEADERS} ${SOURCES})
