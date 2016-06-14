function(do_cuda_compile outfiles)
    if(NOT CUDA_FOUND)
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

if(BUILD_SHARED_LIBS)
    set(VISIONARAY_LIBRARY "${COVISEDIR}/${ARCHSUFFIX}/lib/libvisionaray${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    set(VISIONARAY_LIBRARY "${COVISEDIR}/${ARCHSUFFIX}/lib/libvisionaray${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()
set(VISIONARAY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../3rdparty/visionaray/include")

find_package(Boost COMPONENTS chrono filesystem iostreams system thread REQUIRED)
find_package(CUDA)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenSceneGraph REQUIRED)
if (NOT APPLE AND NOT WIN32)
    find_package(PTHREAD REQUIRED)
endif()

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
include_directories(${VISIONARAY_INCLUDE_DIR})

set(EXTRA_LIBS
    ${EXTRA_LIBS}
    ${Boost_LIBRARIES}
    ${GLEW_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${VISIONARAY_LIBRARY}
)

if (NOT APPLE AND NOT WIN32)
    include_directories(${PTHREAD_INCLUDE_DIR})
    set(EXTRA_LIBS
        ${EXTRA_LIBS}
        ${PTHREAD_LIBRARY}
    )
endif()

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

if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
    do_cuda_compile(CUDA_SOURCES drawable.cu)
else()
    set(SOURCES
        ${SOURCES}
        drawable.cpp
    )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
cover_add_plugin(Visionaray ${HEADERS} ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(Visionaray ${EXTRA_LIBS})
