using(VRML)

SET(HEADERS
  Thyssen.h
  Elevator.h
  Car.h
)
SET(SOURCES
  Thyssen.cpp
  Elevator.cpp
  Car.cpp
)
cover_add_plugin(Thyssen)
TARGET_LINK_LIBRARIES(Thyssen)
