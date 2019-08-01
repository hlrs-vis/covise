SET(HEADERS
  EKU.h
  Cam.h
  Truck.h
  openGA.hpp
  #myGAproblem.hpp
  
)
SET(SOURCES
  EKU.cpp
  Cam.cpp
  Truck.cpp
  #myGAproblem.cpp
  
)
cover_add_plugin(EKU ${HEADERS} ${SOURCES})
