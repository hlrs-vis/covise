using(VRML)

SET(HEADERS
  Simulink.h
  ../../drivingsim/SteeringWheel/UDPComm.h
)
SET(SOURCES
  Simulink.cpp
  ../../drivingsim/SteeringWheel/UDPComm.cpp
)
include_directories("../../drivingsim/SteeringWheel")
cover_add_plugin(Simulink)
TARGET_LINK_LIBRARIES(Simulink)
#COVER_INSTALL_PLUGIN(Simulink)
