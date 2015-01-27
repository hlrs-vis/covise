IF(NOT CUDA_FOUND OR NOT COVISE_USE_CUDA)
  MESSAGE("Skipping OpenCOVER - cuIsoSurface plugin since CUDA was not found!")
  RETURN()
ENDIF()

COVISE_FIND_PACKAGE(CUDPP)

USING(CUDPP GLEW)

INCLUDE_DIRECTORIES(
  $ENV{COVISEDIR}/src/module/hlrs/cuIsoSurfaceUSG
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_SDK_ROOT_DIR}/common/inc
  ${CUDA_SDK_ROOT_DIR}/common/inc/cudpp
)

SET(cuIsoSurface_CUDA_SOURCES
   ${COVISEDIR}/src/module/hlrs/cuIsoSurfaceUSG/cudaEngine.cu
)

SET(HEADERS
  cuIsoSurface.h
)
SET(SOURCES
  cuIsoSurface.cpp
  ${cuIsoSurface_CUDA_SOURCES}
)
ADD_DEFINITIONS(-DCUDA_ISO)
ADD_DEFINITIONS(-DUSE_VBO)
ADD_DEFINITIONS(-DRENDER_STATE)

#IF(CMAKE_COMPILER_IS_GNUCXX)
#   SET(COVISE_COMPILE_FLAGS "${COVISE_COMPILE_FLAGS} -fvisibility=hidden")
#ENDIF()

CUDA_COMPILE(cuIsoSurface_CUDA_OBJ ${cuIsoSurface_CUDA_SOURCES} SHARED OPTIONS "-D__builtin_stdarg_start=__builtin_va_start")
  

cover_add_plugin(cuIsoSurface ${cuIsoSurface_CUDA_OBJ} ${COVISE_DO_LIBRARY} ${COVISE_CORE_LIBRARY} ${COVISE_SHM_LIBRARY} ${COVISE_GPU_LIBRARIES})
