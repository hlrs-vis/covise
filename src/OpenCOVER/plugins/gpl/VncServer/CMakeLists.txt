COVISE_FIND_PACKAGE(LibVncServer)
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
  cuda_compile(CUDA_OBJ ${CUDA_SOURCES} SHARED OPTIONS "-D__builtin_stdarg_start=__builtin_va_start")
endif()
  
set(SOURCES
  ${CUDA_SOURCES}
  VncServer.cpp
)

cover_add_plugin(VncServer ${CUDA_OBJ})
covise_wnoerror(VncServer)

target_link_libraries(VncServer ${EXTRA_LIBS} ${LIBVNCSERVER_SERVER_LIBRARY})
