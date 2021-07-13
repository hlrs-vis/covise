USING(Pthreads)
using(VRML)

if(APPLE)
   return()
endif()

SET(HEADERS
  Skateboard.h
  ../../drivingsim/SteeringWheel/UDPComm.h
)
SET(SOURCES
  Skateboard.cpp
  ../../drivingsim/SteeringWheel/UDPComm.cpp
)
include_directories("../../drivingsim/SteeringWheel")
cover_add_plugin(Skateboard)
TARGET_LINK_LIBRARIES(Skateboard ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Skateboard)
