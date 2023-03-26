using(VRML)
IF(COVISE_BUILD_ONLY_COVER OR NOT CUDA_FOUND OR NOT COVISE_USE_CUDA)
  #MESSAGE("Skipping OpenCOVER - startDust plugin since CUDA was not found")
  RETURN()
ENDIF()

COVISE_FIND_PACKAGE(CUDPP)
COVISE_FIND_PACKAGE(GLEW)

USING(CUDPP)
USING(GLEW)
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

#message("starDust_CUDA_OBJ")
#message(${starDust_CUDA_OBJ})
cover_add_plugin(starDust ${SOURCES})
REMOVE_COVISE_COMPILE_FLAGS(starDust "-wd4251")
REMOVE_COVISE_COMPILE_FLAGS(starDust "-wd4335")
