SET(HEADERS
  EKU.h
  Cam.h
  Truck.h
)
SET(SOURCES
  EKU.cpp
  Cam.cpp
  Truck.cpp
)
cover_add_plugin(EKU ${HEADERS} ${SOURCES})
