USE_VISIONARAY()

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

find_package(OpenSceneGraph REQUIRED)
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
set(EXTRA_LIBS ${EXTRA_LIBS} ${OPENSCENEGRAPH_LIBRARIES})

set(HEADERS
    kernels/bvh_costs_kernel.h
    kernels/normals_kernel.h
    kernels/tex_coords_kernel.h
    drawable.h
    state.h
    visionaray_plugin.h
)

set(SOURCES
    visionaray_plugin.cpp
)

if(COVISE_USE_CUDA AND CUDA_FOUND)
    do_cuda_compile(CUDA_SOURCES drawable.cu)
else()
    set(SOURCES
        ${SOURCES}
        drawable.cpp
    )
endif()

cover_add_plugin(Visionaray ${HEADERS} ${SOURCES} ${CUDA_SOURCES})
