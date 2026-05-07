using(JSBSIM)
using(VRML)
set (CMAKE_CXX_STANDARD 17)

SET(HEADERS
    JSBSim.h
    VrmlNodeThermal.h
)
SET(SOURCES
    JSBSim.cpp
    VrmlNodeThermal.cpp
)
cover_add_plugin(JSBSim coVRGeoData)
#COVER_INSTALL_PLUGIN(JSBSim)
