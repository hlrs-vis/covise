#set(VISIONARAY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../3rdparty/visionaray/include")
#include_directories(${VISIONARAY_INCLUDE_DIR})

USING(GLEW)
USING(VISIONARAY)
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})

function(do_cuda_compile outfiles)
    if(NOT COVISE_USE_CUDA OR NOT CUDA_FOUND)
        return()
    endif()

    foreach(f ${ARGN})
        cuda_compile(cuda_compile_obj
            ${f} SHARED)
        set(out
            ${out}
            ${f}
            ${cuda_compile_obj})
    endforeach()

    set(${outfiles}
        ${out}
        PARENT_SCOPE)
endfunction()

SET(HEADERS
  ColorSphere.h
  PointRayTracerGlobals.h
  PointRayTracerKernel.h
  PointRayTracerPlugin.h
  PointRayTracerDrawable.h
  PointReader.h
  rply.h
)
SET(SOURCES
  PointRayTracerPlugin.cpp
  PointReader.cpp
  rply.c
)

if(COVISE_USE_CUDA AND CUDA_FOUND)
    do_cuda_compile(CUDA_SOURCES PointRayTracerDrawable.cu)
else()
    set(SOURCES
        ${SOURCES}
        PointRayTracerDrawable.cpp
    )
endif()

cover_add_plugin(PointRayTracerPlugin ${HEADERS} ${SOURCES} ${CUDA_SOURCES})
SET_TARGET_PROPERTIES(PointRayTracerPlugin PROPERTIES OUTPUT_NAME PointRayTracer)

