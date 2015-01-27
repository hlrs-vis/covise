message("SPH: please fix me")

COVISE_FIND_PACKAGE(CUDPP)

USING(CUDPP GLEW)
IF(NOT CUDA_FOUND OR NOT COVISE_USE_CUDA)
  MESSAGE("Skipping OpenCOVER - SPH plugin since CUDA was not found!")
  RETURN()
ENDIF()

INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_SDK_ROOT_DIR}/common/inc
  ${CUDA_SDK_ROOT_DIR}/common/inc/cudpp
)

add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

SET(SPH_CUDA_SOURCES
   CudaMax.cu
CudaUtils.cu
SimBase.cu
SimDEM.cu
SimSimpleSPH.cu
SimSnowSPH.cu
UniformGrid.cu
cuPrintf.cu
timer.cu
)

SET(HEADERS
SPH.h
renderParticles.h
osgSimBuffer.h
Config.h
CudaBuffer.h
ParticleData.h
SimBuffer.h
SimBufferCuda.h
SimBufferManager.h
SimCudaAllocator.h
SimCudaHelper.h
SimLogger.h
SimSettings.h
SimulationSystem.h
cutil_gl_error.h
cutil_gl_inline.h
cutil_inline.h
cutil_inline_bankchecker.h
cutil_inline_drvapi.h
cutil_inline_runtime.h
cutil_math.h
timer.h
CudaMax.cuh
CudaUtils.cuh
K_Coloring.cuh
K_Common.cuh
K_SPH_Common.cuh
K_UniformGrid.cuh
SimBase.cuh
SimDEM.cuh
SimSimpleSPH.cuh
SimSnowSPH.cuh
UniformGrid.cuh
cuPrintf.cuh
K_Boundaries_Common.inl
K_Boundaries_Terrain.inl
K_Boundaries_Walls.inl
K_Coloring.inl
K_Common.inl
K_SPH_Common.inl
K_SPH_Kernels.inl
K_SPH_Kernels_cubic.inl
K_SPH_Kernels_gaussian.inl
K_SPH_Kernels_poly6.inl
K_SPH_Kernels_quadratic.inl
K_SPH_Kernels_quartic.inl
K_SPH_Kernels_quintic.inl
K_SPH_Kernels_spiky.inl
K_SPH_Kernels_viscosity.inl
K_SimDEM.inl
K_SimpleSPH.inl
K_SimpleSPH_Integrate.inl
K_SimpleSPH_Step1.inl
K_SimpleSPH_Step2.inl
K_SnowSPH.inl
K_SnowSPH_Integrate.inl
K_SnowSPH_Step1.inl
K_SnowSPH_Step2.inl
K_SnowSPH_Step3.inl
K_UniformGrid_Hash.inl
K_UniformGrid_NeighborList.inl
K_UniformGrid_Update.inl
K_UniformGrid_Utils.inl
)
SET(SOURCES
osgSimBuffer.cpp
renderParticles.cpp
  SimBufferCuda.cpp
SimCudaHelper.cpp
SimulationSystem.cpp
SPH.cpp
  ${SPH_CUDA_SOURCES}
)

CUDA_COMPILE(SPH_CUDA_OBJ ${SPH_CUDA_SOURCES} SHARED OPTIONS "-D__builtin_stdarg_start=__builtin_va_start")
  
#message("SPH_CUDA_OBJ")
#message(${SPH_CUDA_OBJ})
cover_add_plugin(SPH ${SPH_CUDA_OBJ} coVRML)
