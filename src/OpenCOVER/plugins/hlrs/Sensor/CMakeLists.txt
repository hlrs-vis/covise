USING(Pthreads)
USING(VRML)

if(APPLE)
   return()
endif()

SET(HEADERS
  Sensor.h
  ../../drivingsim/SteeringWheel/UDPComm.h
)
SET(SOURCES
  Sensor.cpp
  ../../drivingsim/SteeringWheel/UDPComm.cpp
)
include_directories("../../drivingsim/SteeringWheel")
cover_add_plugin(Sensor)
TARGET_LINK_LIBRARIES(Sensor ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Sensor)

