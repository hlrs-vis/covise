SET(HEADERS
  EKU.h
  Cam.h
  Truck.h
  openGA.hpp
)
SET(SOURCES
  EKU.cpp
  Cam.cpp
  Truck.cpp
  Optimize.cpp
)
cover_add_plugin(EKU ${HEADERS} ${SOURCES})
