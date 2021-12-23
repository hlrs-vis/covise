
set(VRB_HEADER 
vrbClientInterface.h
)

set(VRB_SOURCES 
vrbClientInterface.cpp
)


set(targetname coVRBClientInterface)
ADD_COVISE_LIBRARY(${targetname} ${VRB_SOURCES} ${VRB_HEADER})
TARGET_LINK_LIBRARIES(${targetname} PRIVATE coVRBClient)
COVISE_INSTALL_TARGET(${targetname})
COVISE_INSTALL_HEADERS(${targetname} ${VRB_HEADER})

# ADD_LIBRARY(${targetname}_extra test.cpp test.h)
# TARGET_LINK_LIBRARIES(${targetname}_extra PRIVATE ${targetname})
# vistle_export_library(${targetname}_extra ${VRB_HEADER})
