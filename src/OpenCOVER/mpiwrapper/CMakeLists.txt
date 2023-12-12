USING(MPI optional)
USING(PTHREAD)
IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()

include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

set(MPIWRAPPER_HEADERS mpicover.h)
add_covise_library(mpicover SHARED mpicover.cpp ${MPIWRAPPER_HEADERS})
target_include_directories(mpicover PUBLIC ${PTHREAD_INCLUDE_DIR})
target_link_libraries(mpicover coOpenCOVER ${COVISE_VRBCLIENT_LIBRARY}
   ${COVISE_GRMSG_LIBRRARY} ${COVISE_UTIL_LIBRARY} ${EXTRA_LIBS})
qt_use_modules(mpicover Core)
covise_install_target(mpicover)
covise_install_headers(mpiwrapper ${MPIWRAPPER_HEADERS})
