covise_find_package(Protobuf CONFIG)
covise_find_package(gRPC CONFIG)

if (NOT TARGET protobuf::libprotobuf)
    return()
endif()
if (NOT TARGET gRPC::grpc)
    return()
endif()
if (NOT TARGET gRPC::grpc++)
    return()
endif()

set(PROTO_FILES
    aural-reality-protocol/audio_file.proto
    aural-reality-protocol/auralreality.proto
    aural-reality-protocol/listener.proto
    aural-reality-protocol/playback_config.proto
    aural-reality-protocol/quaternion.proto
    aural-reality-protocol/rotation.proto
    aural-reality-protocol/source.proto
    aural-reality-protocol/speaker.proto
    aural-reality-protocol/tmt_service.proto
    aural-reality-protocol/track.proto
    aural-reality-protocol/trajectory.proto
    aural-reality-protocol/transform.proto
    aural-reality-protocol/utils.proto
    aural-reality-protocol/vector.proto
)

SET(HEADERS
    AuralRealityPlugin.h
)

SET(SOURCES
    AuralRealityPlugin.cpp
)

LIST(APPEND EXTRA_LIBS 
    protobuf::libprotobuf 
    gRPC::grpc 
    gRPC::grpc++
)

cover_add_plugin(AuralReality ${HEADERS} ${SOURCES})

get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(
    TARGET AuralReality 
    IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/aural-reality-protocol"
    LANGUAGE cpp
    PROTOS ${PROTO_FILES}
)
protobuf_generate(
    TARGET AuralReality 
    LANGUAGE grpc 
    PROTOS ${PROTO_FILES}
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc 
    IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/aural-reality-protocol"
    PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}"
)
