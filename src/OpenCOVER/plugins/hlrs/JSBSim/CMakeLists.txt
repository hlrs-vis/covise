using(JSBSIM)
using(VRML)
USING(PROJ4)
set (CMAKE_CXX_STANDARD 17)

SET(HEADERS
    ../../drivingsim/SteeringWheel/UDPComm.h
    JSBSim.h
)
SET(SOURCES
    ../../drivingsim/SteeringWheel/UDPComm.cpp
    JSBSim.cpp
)
include_directories("../../drivingsim/SteeringWheel")
cover_add_plugin(JSBSim)
TARGET_LINK_LIBRARIES(JSBSim coRSClient)
#COVER_INSTALL_PLUGIN(JSBSim)
