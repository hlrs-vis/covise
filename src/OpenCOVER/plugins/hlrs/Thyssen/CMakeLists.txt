using(VRML)

SET(HEADERS
  Thyssen.h
  Elevator.h
  Exchanger.h
  Car.h
)
SET(SOURCES
  Thyssen.cpp
  Elevator.cpp
  Exchanger.cpp
  Car.cpp
)
cover_add_plugin(Thyssen)
TARGET_LINK_LIBRARIES(Thyssen)
