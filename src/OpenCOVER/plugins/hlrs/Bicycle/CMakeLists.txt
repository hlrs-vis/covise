USING(LibUSB1)
USING(Pthreads)
using(VRML)

if(APPLE)
   return()
endif()

SET(HEADERS
  AVRserialcom.h
  Tacx.h
  Bicycle.h
  FlightGear.h
  ../../drivingsim/SteeringWheel/UDPComm.h
)
SET(SOURCES
  AVRserialcom.cpp
  Tacx.cpp
  Bicycle.cpp
  FlightGear.cpp
  ../../drivingsim/SteeringWheel/UDPComm.cpp
)
include_directories("../../drivingsim/SteeringWheel")
cover_add_plugin(Bicycle)
TARGET_LINK_LIBRARIES(Bicycle ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Bicycle)

SET(HEADERS
  Tacx.h
  ../../drivingsim/SteeringWheel/UDPComm.h
)
SET(SOURCES
  TacxServer.cpp
  Tacx.cpp
  ../../drivingsim/SteeringWheel/UDPComm.cpp
)

ADD_COVISE_EXECUTABLE(TacxServer)

TARGET_LINK_LIBRARIES(TacxServer ${EXTRA_LIBS} coNet coUtil coConfig )
