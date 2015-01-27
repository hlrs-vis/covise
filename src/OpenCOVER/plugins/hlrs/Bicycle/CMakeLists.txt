USING(LibUSB)
USING(Pthreads)

if(APPLE)
   return()
endif()

SET(HEADERS
  AVRserialcom.h
  Tacx.h
  Bicycle.h
  ../../drivingsim/SteeringWheel/UDPComm.h
)
SET(SOURCES
  AVRserialcom.cpp
  Tacx.cpp
  Bicycle.cpp
  ../../drivingsim/SteeringWheel/UDPComm.cpp
)
include_directories("../../drivingsim/SteeringWheel")
cover_add_plugin(Bicycle)
TARGET_LINK_LIBRARIES(Bicycle coVRML ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Bicycle)

