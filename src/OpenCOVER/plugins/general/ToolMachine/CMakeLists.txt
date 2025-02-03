
# using(OPEN62541)
using(VRML)

if(NOT open62541_FOUND)
     MESSAGE("open62541 not found, not building ToolMachine")
    return()
endif()


SET(HEADERS
../CsvPointCloud/OctUtils.h
Currents.h
MathExpressions.h
Oct.h
Plugin.h
Tool.h
ToolChanger/Arm.h
ToolChanger/Changer.h
ToolChanger/Tool.h
ToolChanger/ToolChanger.h
ToolChanger/Utility.h
ToolMachine.h
VrmlMachine.h
VrmlToolChanger.h
)

SET(SOURCES
../CsvPointCloud/OctUtils.cpp
Currents.cpp
MathExpressions.cpp
Oct.cpp
Plugin.cpp
Tool.cpp
ToolChanger/Arm.cpp
ToolChanger/Changer.cpp
ToolChanger/Tool.cpp
ToolChanger/ToolChanger.cpp
ToolChanger/Utility.cpp
ToolMachine.cpp
VrmlMachine.cpp
VrmlToolChanger.cpp
)

cover_add_plugin(ToolMachine)
target_link_libraries(ToolMachine coOpenOpcUaClient exprtk coPointerUtil)

