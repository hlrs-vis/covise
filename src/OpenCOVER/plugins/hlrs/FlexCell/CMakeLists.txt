using(VRML)

covise_find_package(rabbitmq-c)

if(NOT rabbitmq-c_FOUND)
  message(STATUS "rabbitmq-c not found, skipping RabbitMQ DataClient")
    return()
endif()

set(HEADERS
    FlexCell.h
    VrmlNodes.h
)

set(SOURCES
    FlexCell.cpp
    VrmlNodes.cpp
)

cover_add_plugin(FlexCell)
target_link_libraries(FlexCell coOpenOpcUaClient coPointerUtil nlohmann_json::nlohmann_json  rabbitmq::rabbitmq)
