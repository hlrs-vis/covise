find_package(Protobuf CONFIG)
find_package(cppzmq)

if (NOT TARGET protobuf::libprotobuf)
    return()
endif()

if (NOT TARGET cppzmq)
    return()
endif()

SET(HEADERS
    AuralRealityPlugin.h
    CustomTransformInteractor.h
    Speaker.h
    Selection.h
    Trajectory.h
    ZeroMqRpcChannel.h
)

SET(SOURCES
    AuralRealityPlugin.cpp
    CustomTransformInteractor.cpp
    Speaker.cpp
    Selection.cpp
    Trajectory.cpp
    ZeroMqRpcChannel.cpp
)

add_subdirectory(aural-reality-protocol/arrpc)

# set_target_properties(arrpc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
COVISE_INSTALL_TARGET(arrpc)

###
### Now follows a copy-paste of the cover_add_plugin macro, because we need to link 
### the static library as PRIVATE, and the rest as PUBLIC, and the macro does not
### support any of that.
###

IF(NOT OPENSCENEGRAPH_FOUND)
    RETURN()
ENDIF()

IF(WIN32)
    ADD_DEFINITIONS(-DIMPORT_PLUGIN)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(SYSTEM
    ${ZLIB_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${OPENSCENEGRAPH_INCLUDE_DIRS}
)
INCLUDE_DIRECTORIES(
    "${COVISEDIR}/src/OpenCOVER"
)
IF(APPLE)
    ADD_LIBRARY(AuralReality MODULE ${SOURCES} ${HEADERS})
ELSE(APPLE)
    ADD_LIBRARY(AuralReality SHARED ${SOURCES} ${HEADERS})
ENDIF(APPLE)

IF("${MAIN_FOLDER}" STREQUAL "")
    set_target_properties(AuralReality PROPERTIES FOLDER "Plugins/${PLUGIN_CATEGORY}")
ELSE()
    set_target_properties(AuralReality PROPERTIES FOLDER "${MAIN_FOLDER}/Plugins/${PLUGIN_CATEGORY}")
ENDIF()

# SET_TARGET_PROPERTIES(AuralReality PROPERTIES PROJECT_LABEL "AuralReality")
SET_TARGET_PROPERTIES(AuralReality PROPERTIES OUTPUT_NAME "AuralReality")

COVISE_ADJUST_OUTPUT_DIR(AuralReality "OpenCOVER/plugins")

# set additional COVISE_COMPILE_FLAGS
SET_TARGET_PROPERTIES(AuralReality PROPERTIES COMPILE_FLAGS "${COVISE_COMPILE_FLAGS}")
# set additional COVISE_LINK_FLAGS
SET_TARGET_PROPERTIES(AuralReality PROPERTIES LINK_FLAGS "${COVISE_LINK_FLAGS}")
# switch off "lib" prefix for MinGW
IF(MINGW)
    SET_TARGET_PROPERTIES(AuralReality PROPERTIES PREFIX "")
ENDIF(MINGW)

TARGET_LINK_LIBRARIES(AuralReality PUBLIC coOpenPluginUtil coOpenCOVER coOpenVRUI coOSGVRUI ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_CONFIG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES})

IF (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    ADD_COVISE_LINK_FLAGS(AuralReality "-Wl,--no-undefined")
ENDIF()

UNSET(SOURCES)
UNSET(HEADERS)
target_compile_definitions(AuralReality PRIVATE COVER_PLUGIN_NAME="AuralReality")
# IF(COVISE_USE_QT)
#     qt_use_modules(AuralReality Core)
# ENDIF(COVISE_USE_QT)

# TARGET_LINK_LIBRARIES(AuralReality PUBLIC ${LIBS})
IF(DEFINED EXTRA_LIBS)
    TARGET_LINK_LIBRARIES(AuralReality PUBLIC ${EXTRA_LIBS})
ENDIF()

if(CUDA_FOUND AND COVISE_USE_CUDA)
    TARGET_LINK_LIBRARIES(AuralReality PUBLIC ${CUDA_LIBRARIES})
endif()

COVER_INSTALL_PLUGIN(AuralReality)

target_link_libraries(AuralReality PRIVATE arrpc )
# target_include_directories(AuralReality
#     PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/aural-reality-protocol/arrpc/include"
# )

target_link_libraries(AuralReality PUBLIC protobuf::libprotobuf cppzmq)
