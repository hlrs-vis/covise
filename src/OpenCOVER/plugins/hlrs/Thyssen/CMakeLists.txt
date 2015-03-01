using(VRML)

SET(HEADERS
  Thyssen.h
  Elevator.h
)
SET(SOURCES
  Thyssen.cpp
  Elevator.cpp
)
cover_add_plugin(Thyssen)
TARGET_LINK_LIBRARIES(Thyssen)
