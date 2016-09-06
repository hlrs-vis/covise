using(VRML)
COVISE_FIND_PACKAGE(CUDPP)

USING(CUDPP GLEW)
IF(NOT CUDA_FOUND OR NOT COVISE_USE_CUDA)
  MESSAGE("Skipping OpenCOVER - cudaLight plugin since CUDA was not found!")
  RETURN()
ENDIF()

INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_SDK_ROOT_DIR}/common/inc
  ${CUDA_SDK_ROOT_DIR}/common/inc/cudpp
)

SET(cudaLight_CUDA_SOURCES
   matrixLightCuda.cu
)

SET(HEADERS
  cudaLight.h
  renderTexture.h
)
SET(SOURCES
  cudaLight.cpp
  renderTexture.cpp
  ${cudaLight_CUDA_SOURCES}
)

CUDA_COMPILE(cudaLight_CUDA_OBJ ${cudaLight_CUDA_SOURCES} SHARED OPTIONS "-D__builtin_stdarg_start=__builtin_va_start")
  
#message("cudaLight_CUDA_OBJ")
#message(${cudaLight_CUDA_OBJ})
cover_add_plugin(cudaLight ${cudaLight_CUDA_OBJ})
