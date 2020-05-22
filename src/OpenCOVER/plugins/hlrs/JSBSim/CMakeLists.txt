using(JSBSIM)

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
TARGET_LINK_LIBRARIES(JSBSim)
#COVER_INSTALL_PLUGIN(JSBSim)
