set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nvtop/cmake/modules;${CMAKE_MODULE_PATH}")

covise_find_package(NVML)
if (NOT NVML_FOUND)
    return()
endif()

include_directories(nvtop/include)
add_definitions(-D_POSIX_C_SOURCE=199309L)
add_definitions(-D_XOPEN_SOURCE=800)

include_directories(SYSTEM ${NVML_INCLUDE_DIRS})

set(HEADERS
    NvTop.h
)

set(SOURCES
    NvTop.cpp
    nvtop/src/extract_gpuinfo.c
    nvtop/src/get_process_info_linux.c
)

cover_add_plugin(NvTop)
target_link_libraries(NvTop ${NVML_LIBRARIES})
