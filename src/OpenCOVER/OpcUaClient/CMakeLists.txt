if(NOT open62541_FOUND)
     MESSAGE("open62541 not found, not building OpenOpcUaClient library")
    return()
endif()

SET(HEADERS
    ${HEADERS}
    export.h
    observerHandle.h
    opcua.h
    opcuaClient.h
    types.h
    uaVariantPtr.h
    variantAccess.h
)

SET(SOURCES
${SOURCES}    
    observerHandle.cpp
    opcua.cpp
    opcuaClient.cpp
    uaVariantPtr.cpp
    variantAccess.cpp
)

ADD_COVISE_LIBRARY(coOpenOpcUaClient)

TARGET_LINK_LIBRARIES(coOpenOpcUaClient coOpenCOVER ${OPENSCENEGRAPH_LIBRARIES} open62541::open62541)
COVISE_INSTALL_TARGET(coOpenOpcUaClient)
COVISE_INSTALL_HEADERS(coOpenOpcUaClient ${HEADERS})

TARGET_INCLUDE_DIRECTORIES(coOpenOpcUaClient PRIVATE ${OPENSCENEGRAPH_INCLUDE_DIRS})

COVISE_WNOERROR(coOpenOpcUaClient BASEARCH zebu)