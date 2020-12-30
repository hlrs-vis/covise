

set(VRB_TEST_SOURCES 
    main.cpp
    message_macros.cpp
    testObject.cpp
    tryPrint.cpp
    coSpawnProgram.cpp
	CRB_EXEC.cpp
)

set(VRB_TEST_HEADER 
    message_macros.h
    testObject.h
    tryPrint.h
    coSpawnProgram.h
	CRB_EXEC.h
)

ADD_COVISE_EXECUTABLE(VrbTests ${VRB_TEST_SOURCES} ${VRB_TEST_HEADER})
target_link_libraries(VrbTests coVRBClient)
target_include_directories(VrbTests PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/kernel>
)

ADD_DEFINITIONS(-DHAVE_COVISE)

