using(osgcal)

find_package(cppzmq)
find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS traffic_simulator.proto) 

set(HEADERS
    Traffic.h
    Geometry.h
    PedestrianGeometry.h
    CarGeometry.h
    Connector.h
    ConnectorZmq.h
    sumo/ConnectorSumo.h
    sumo/socket.h
    sumo/storage.h
    sumo/TraCIAPI.h
    sumo/TraCIDefs.h
    sumo/TraCIConstants.h
    sumo/config.h
)
set(SOURCES
    Traffic.cpp
    Geometry.cpp
    PedestrianGeometry.cpp
    CarGeometry.cpp
    ConnectorZmq.cpp
    sumo/ConnectorSumo.cpp
    sumo/socket.cpp
    sumo/storage.cpp
    sumo/TraCIAPI.cpp
)

INCLUDE_DIRECTORIES(
    ${OSGCAL_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
)

# Fix protobuf linking. Thanks, mvf!
# https://github.com/protocolbuffers/protobuf/issues/12637#issuecomment-1871458639
string(REGEX REPLACE "^[0-9]+\.([0-9]+\.[0-9]+)$" "\\1.0" proto_libver "${Protobuf_VERSION}")
if(proto_libver VERSION_GREATER_EQUAL "22")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(protobuf REQUIRED IMPORTED_TARGET protobuf=${proto_libver})
    set(PROTO_LIB PkgConfig::protobuf)
else()
    set(PROTO_LIB protobuf::libprotobuf)
endif()

cover_add_plugin(Traffic ${HEADERS} ${PROTO_HDRS} ${SOURCES} ${PROTO_SRCS})

target_link_libraries(Traffic 
    ${EXTRA_LIBS}
    ${OSGCAL_LIBRARIES}
    cppzmq
    ${PROTO_LIB}
)
 
