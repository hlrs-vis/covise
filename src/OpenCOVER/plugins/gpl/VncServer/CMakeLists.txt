COVISE_FIND_PACKAGE(LIBVNCSERVER)
if(NOT LIBVNCSERVER_FOUND)
   RETURN()
endif()

using(GLEW)
include_directories(${GLEW_INCLUDE_DIR} ${LIBVNCSERVER_INCLUDE_DIR})

set(EXTRA_LIBS ${EXTRA_LIBS} ${GLEW_LIBRARY})
  
set(HEADERS
  VncServer.h
  ReadBackCuda.h
)

IF(CUDA_FOUND AND COVISE_USE_CUDA)
  set(CUDA_SOURCES
      ReadBackCuda.cu
  )

  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
endif()
  
set(SOURCES
  ${CUDA_SOURCES}
  VncServer.cpp
)

cover_add_plugin(VncServer ${SOURCES})
covise_wnoerror(VncServer)

target_link_libraries(VncServer ${EXTRA_LIBS} ${LIBVNCSERVER_SERVER_LIBRARY})
