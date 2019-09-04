SET(HEADERS
  EKU.h
  Cam.h
  Truck.h
  openGA.hpp
  GA.hpp
  FileReader.hpp
  Sensor.h
  
)
SET(SOURCES
  EKU.cpp
  Cam.cpp
  Truck.cpp
  GA.cpp
  FileReader.cpp
  Sensor.cpp
  
)
cover_add_plugin(EKU ${HEADERS} ${SOURCES})
