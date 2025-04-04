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

#message("cudaLight_CUDA_OBJ")
#message(${cudaLight_CUDA_OBJ})
cover_add_plugin(cudaLight ${SOURCES})

REMOVE_COVISE_COMPILE_FLAGS(cudaLight "-wd4251")
REMOVE_COVISE_COMPILE_FLAGS(cudaLight "-wd4335")