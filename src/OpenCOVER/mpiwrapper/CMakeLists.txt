USING(MPI)
USING(VTK optional)

IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()

include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

ADD_DEFINITIONS(-DRENAME_MAIN=realmain)

IF(APPLE)
   SET(EXTRA_LIBS ${EXTRA_LIBS} "-framework AppKit")
   set(SOURCES ${SOURCES} ../main/osx-util.mm)
ENDIF(APPLE)

covise_add_library(mpicover MODULE ../main/OpenCOVERMain.cpp)
target_link_libraries(mpicover coOpenCOVER ${COVISE_VRBCLIENT_LIBRARY}
   ${COVISE_GRMSG_LIBRRARY} ${COVISE_UTIL_LIBRARY} ${EXTRA_LIBS})
covise_install_target(mpicover)
