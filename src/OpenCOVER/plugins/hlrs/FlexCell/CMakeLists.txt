using(VRML)

set(HEADERS
    FlexCell.h
    VrmlNodes.h
)

set(SOURCES
    FlexCell.cpp
    VrmlNodes.cpp
)

if(TARGET coOpenOpcUaClient AND TARGET coPointerUtil)
    cover_add_plugin(FlexCell)
    target_link_libraries(FlexCell coOpenOpcUaClient coPointerUtil nlohmann_json::nlohmann_json)
    add_dependencies(FlexCell coOpenOpcUaClient coPointerUtil)
else()
    message(STATUS "coOpenOpcUaClient or coPointerUtil not found - skipping FlexCell build")
endif()
