using(VRML)

SET(HEADERS
  XElevator.h
  XLanding.h
  XCar.h
)
SET(SOURCES
  XElevator.cpp
  XLanding.cpp
  XCar.cpp
)
cover_add_plugin(XElevator)
TARGET_LINK_LIBRARIES(XElevator)
