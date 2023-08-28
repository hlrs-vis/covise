
# using(OPEN62541)
using(VRML)

if(NOT open62541_FOUND)
    return()
endif()


SET(HEADERS
ToolMachine.h
)

SET(SOURCES
ToolMachine.cpp
)

cover_add_plugin(ToolMachine)
target_link_libraries(ToolMachine)

