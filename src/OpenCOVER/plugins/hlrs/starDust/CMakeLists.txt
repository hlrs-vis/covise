using(VRML)
COVISE_FIND_PACKAGE(CUDPP)

USING(CUDPP GLEW)
IF(NOT CUDA_FOUND OR NOT COVISE_USE_CUDA)
  MESSAGE("Skipping OpenCOVER - starDust plugin since CUDA was not found!")
  RETURN()
ENDIF()

INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_SDK_ROOT_DIR}/common/inc
  ${CUDA_SDK_ROOT_DIR}/common/inc/cudpp
)

SET(starDust_CUDA_SOURCES
   bodysystemcuda.cu
)

SET(HEADERS
  starDust.h
  renderParticles.h
  bodysystem.h
  solarSystemNode.h
)
SET(SOURCES
  starDust.cpp
  renderParticles.cpp
  solarSystemNode.cpp
  ${starDust_CUDA_SOURCES}
)

CUDA_COMPILE(starDust_CUDA_OBJ ${starDust_CUDA_SOURCES} SHARED OPTIONS "-D__builtin_stdarg_start=__builtin_va_start")
  
#message("starDust_CUDA_OBJ")
#message(${starDust_CUDA_OBJ})
cover_add_plugin(starDust ${starDust_CUDA_OBJ})
