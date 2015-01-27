IF(NOT CUDA_FOUND OR NOT COVISE_USE_CUDA)
  MESSAGE("Skipping OpenCOVER - cuCuttingSurface plugin since CUDA was not found")
  RETURN()
ENDIF()

COVISE_FIND_PACKAGE(CUDPP)
COVISE_FIND_PACKAGE(GLEW)

USING(CUDPP)
USING(GLEW)

INCLUDE_DIRECTORIES(
  $ENV{COVISEDIR}/src/module/hlrs/cuIsoSurfaceUSG
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_SDK_ROOT_DIR}/common/inc
  ${CUDA_SDK_ROOT_DIR}/common/inc/cudpp
)

SET(cuCuttingSurface_CUDA_SOURCES
  ${COVISEDIR}/src/module/hlrs/cuIsoSurfaceUSG/cudaEngine.cu
)

SET(HEADERS
  cuCuttingSurface.h
)
SET(SOURCES
  cuCuttingSurface.cpp
  ${cuCuttingSurface_CUDA_SOURCES}
)

ADD_DEFINITIONS(-DUSE_VBO)
ADD_DEFINITIONS(-DRENDER_STATE)
#ADD_DEFINITIONS(-DLIC)

CUDA_COMPILE(cuCuttingSurface_CUDA_OBJ ${cuCuttingSurface_CUDA_SOURCES} SHARED OPTIONS "-D__builtin_stdarg_start=__builtin_va_start")

cover_add_plugin(cuCuttingSurface ${cuCuttingSurface_CUDA_OBJ} ${COVISE_DO_LIBRARY} ${COVISE_SHM_LIBRARY} ${COVISE_CONFIG_LIBRARY})
#TARGET_LINK_LIBRARIES(cuCuttingSurface coOpenVRUI coDo coCore ${EXTRA_LIBS})
