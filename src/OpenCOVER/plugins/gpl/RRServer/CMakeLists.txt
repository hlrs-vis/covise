COVISE_FIND_PACKAGE(JPEGTURBO)
IF(NOT JPEGTURBO_FOUND)
   RETURN()
ENDIF(NOT JPEGTURBO_FOUND)
if(NOT TURBOJPEG_LIBRARY_STATIC)
   return()
endif()
if(NOT TURBOJPEG_HAVE_TJMCUWIDTH)
   return()
endif()
if (NOT TURBOJPEG_HAVE_INTERNAL)
   return()
endif()
using(GLEW)

COVISE_FIND_PACKAGE(X11)
if(NOT X11_FOUND)
   return()
endif()

add_definitions(-DRR_COVISE)

include_directories(${GLEW_INCLUDE_DIR} ${JPEGTURBO_INCLUDE_DIR} ${X11_INCLUDE_DIR})

set(EXTRA_LIBS ${EXTRA_LIBS} ${TURBOJPEG_LIBRARIES} ${GLEW_LIBRARY})
  
include_directories(vgl/include vgl/rr vgl/util)

set(HEADERS
  RRServer.h
  rrxevent.h
  tjplanar.h
  vgl/include/fbx.h
  vgl/include/genericQ.h
  vgl/include/rrerror.h
  vgl/include/rrlog.h
  vgl/include/rrmutex.h
  vgl/include/rrprofiler.h
  vgl/include/rrsocket.h
  vgl/include/rrthread.h
  vgl/include/rrtimer.h
  vgl/include/rrutil.h
  vgl/include/x11err.h
  vgl/rr/fakerconfig.h
  vgl/rr/rrcwin.h
  vgl/rr/rrdisplayclient.h
  vgl/rr/rrframe.h
  vgl/rr/rrglframe.h
  vgl/rr/rr.h
  vgl/rr/vgllogo.h
  vgl/rr/xdk-sym.h
)

IF(CUDA_FOUND AND COVISE_USE_CUDA)
   add_definitions(-DHAVE_CUDA)
   include_directories(${CUDA_INCLUDE_DIRS})
   set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
endif()
  
set(SOURCES
  ${CUDA_SOURCES}
  RRServer.cpp
  tjplanar.c
  vgl/util/rrlog.cpp
  vgl/util/rrsocket.cpp
  vgl/util/genericQ.cpp
  vgl/rr/rrframe.cpp
  vgl/rr/rrdisplayclient.cpp
  vgl/rr/fakerconfig.cpp
)

cover_add_plugin(RRServer ${CUDA_OBJ} ${HEADER} ${SOURCES})
covise_wnoerror(RRServer)

target_link_libraries(RRServer ${EXTRA_LIBS} ${COVISE_RHR_LIBRARY})
