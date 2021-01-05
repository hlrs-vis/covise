

set(VRB_TEST_SOURCES 
    CRB_EXEC.cpp
    coSpawnProgram.cpp
    main.cpp
    message_macros.cpp
    testObject.cpp
    tokenbuffer_util.cpp
    tryPrint.cpp
)

set(VRB_TEST_HEADER 
    CRB_EXEC.h
    coSpawnProgram.h
    message_macros.h
    testObject.h
    tokenbuffer_util.h
    tryPrint.h
)

ADD_COVISE_EXECUTABLE(VrbTests ${VRB_TEST_SOURCES} ${VRB_TEST_HEADER})
target_link_libraries(VrbTests coVRBClient coMessages)
target_include_directories(VrbTests PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/kernel>
)

ADD_DEFINITIONS(-DHAVE_COVISE)

