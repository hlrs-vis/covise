SET(HEADERS
 Profiling.h
 Helper.h
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
)


cover_add_plugin(SensorPlacement ${HEADERS} ${SOURCES})