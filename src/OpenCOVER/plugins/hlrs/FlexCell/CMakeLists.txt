using(VRML)

set(HEADERS
    FlexCell.h
    VrmlNodes.h
)

set(SOURCES
    FlexCell.cpp
    VrmlNodes.cpp
)

cover_add_plugin(FlexCell)
target_link_libraries(FlexCell nlohmann_json::nlohmann_json coDataClient)
