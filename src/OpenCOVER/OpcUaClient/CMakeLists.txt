if(NOT open62541_FOUND)
     MESSAGE("open62541 not found, not building OpenOpcUaClient library")
    return()
endif()

SET(HEADERS
    export.h
    opcua.h
    opcuaClient.h
    types.h
    uaVariantPtr.h
    )
    
SET(SOURCES
    opcua.cpp
    opcuaClient.cpp
    types.cpp
    uaVariantPtr.cpp
)

ADD_COVISE_LIBRARY(coOpenOpcUaClient)

TARGET_LINK_LIBRARIES(coOpenOpcUaClient coOpenCOVER open62541::open62541 coDataClient)
COVISE_INSTALL_TARGET(coOpenOpcUaClient)
COVISE_INSTALL_HEADERS(coOpenOpcUaClient ${HEADERS})

COVISE_WNOERROR(coOpenOpcUaClient BASEARCH zebu)