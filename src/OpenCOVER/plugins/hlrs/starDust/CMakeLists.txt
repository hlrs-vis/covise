using(VRML)
USING(CUDA GLEW)
IF(COVISE_BUILD_ONLY_COVER OR NOT CUDA_FOUND OR NOT COVISE_USE_CUDA)
  MESSAGE("Skipping OpenCOVER - startDust plugin since CUDA was not found")
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

cover_add_plugin(starDust ${SOURCES})
REMOVE_COVISE_COMPILE_FLAGS(starDust "-wd4251")
REMOVE_COVISE_COMPILE_FLAGS(starDust "-wd4335")
