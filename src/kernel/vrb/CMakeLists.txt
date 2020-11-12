set(VRB_SOURCES
    SessionID.cpp	
    UserInfo.cpp
    Registry.cpp
    RegistryClass.cpp
    RegistryVariable.cpp
    
)

set(VRB_HEADER
    SessionID.h
    UserInfo.h
    Registry.h
    RegistryClass.h
    RegistryVariable.h
)

ADD_COVISE_LIBRARY(coVRB ${VRB_SOURCES} ${VRB_HEADER})
target_link_libraries(coVRB
                        coNet
                        coUtil 
)
COVISE_INSTALL_TARGET(coVRB)
COVISE_INSTALL_HEADERS(coVRB ${VRB_HEADER})

ADD_SUBDIRECTORY(client)
ADD_SUBDIRECTORY(server)
