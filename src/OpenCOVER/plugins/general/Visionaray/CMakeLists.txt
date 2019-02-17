USING(Visionaray)
USING(TBB)
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

set(HEADERS
    kernels/bvh_costs_kernel.h
    kernels/normals_kernel.h
    kernels/tex_coords_kernel.h
    common.h
    mask_intersector.h
    pathtrace_cpu.h
    pathtrace_gpu.h
    render_cpu.h
    render_gpu.h
    renderer.h
    scene_monitor.h
    state.h
    visionaray_plugin.h
)

set(SOURCES
    pathtrace_cpu.cpp
    render_cpu.cpp
    scene_monitor.cpp
    visionaray_plugin.cpp
)

if(COVISE_USE_CUDA AND CUDA_FOUND)
    do_cuda_compile(
        CUDA_SOURCES
        pathtrace_gpu.cu
        render_gpu.cu
        renderer.cu)
else()
    set(SOURCES
        ${SOURCES}
        renderer.cpp
    )
endif()

cover_add_plugin(Visionaray ${HEADERS} ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(Visionaray ${OPENSCENEGRAPH_LIBRARIES})
target_link_libraries(Visionaray coOpenPluginUtil)
