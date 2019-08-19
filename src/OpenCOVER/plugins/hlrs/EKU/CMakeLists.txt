SET(HEADERS
  EKU.h
  Cam.h
  Truck.h
  openGA.hpp
  GA.hpp
  
)
SET(SOURCES
  EKU.cpp
  Cam.cpp
  Truck.cpp
  GA.cpp
  
)
cover_add_plugin(EKU ${HEADERS} ${SOURCES})
