
# using(OPEN62541)
using(VRML)
if(NOT open62541_FOUND)
    return()
endif()
SET(HEADERS
ToolMachine.h
opcua.h
)

SET(SOURCES
ToolMachine.cpp
opcua.cpp
)

cover_add_plugin(ToolMachine)
target_link_libraries(ToolMachine open62541::open62541)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_covise_compile_flags(ToolMachine "-Wno-narrowing")
endif()
