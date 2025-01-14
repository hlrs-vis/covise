USING(MPI optional)
USING(vsg)
USING(PTHREAD)

include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

set(MPIWRAPPER_HEADERS mpivive.h)
add_covise_library(mpivive SHARED mpivive.cpp ${MPIWRAPPER_HEADERS})
target_include_directories(mpivive PUBLIC ${PTHREAD_INCLUDE_DIR})
target_link_libraries(mpivive vvCore ${COVISE_VRBCLIENT_LIBRARY}
   ${COVISE_GRMSG_LIBRRARY} ${COVISE_UTIL_LIBRARY} ${EXTRA_LIBS})

if(covise_use_qt)
    qt_use_modules(mpivive Core)
endif()

covise_install_target(mpivive)
covise_install_headers(mpiwrapper ${MPIWRAPPER_HEADERS})
