

set(VRB_TEST_SOURCES 
    CRB_EXEC.cpp
    asyncVrb.cpp
    coSpawnProgram.cpp
    connection.cpp
    main.cpp
    messageExchange.cpp
    message_macros.cpp
    syncVar.cpp
    testObject.cpp
    tryPrint.cpp
)

set(VRB_TEST_HEADER 
    CRB_EXEC.h
    asyncVrb.h
    coSpawnProgram.h
    connection.h
    messageExchange.h
    message_macros.h
    syncVar.h
    testObject.h
    tryPrint.h
)

ADD_COVISE_EXECUTABLE(VrbTests ${VRB_TEST_SOURCES} ${VRB_TEST_HEADER})
target_link_libraries(VrbTests coVRBClient coMessages coUtil)
target_include_directories(VrbTests PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/kernel>
)

ADD_DEFINITIONS(-DHAVE_COVISE)

