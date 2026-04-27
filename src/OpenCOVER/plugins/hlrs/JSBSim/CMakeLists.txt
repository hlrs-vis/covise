using(JSBSIM)
using(VRML)
USING(PROJ)
set (CMAKE_CXX_STANDARD 17)

SET(HEADERS
    JSBSim.h
    VrmlNodeThermal.h
)
SET(SOURCES
    JSBSim.cpp
    VrmlNodeThermal.cpp
)
cover_add_plugin(JSBSim)
#COVER_INSTALL_PLUGIN(JSBSim)
