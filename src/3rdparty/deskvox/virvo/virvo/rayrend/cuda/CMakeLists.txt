if(DESKVOX_USE_CUDA)

find_package(CUDA)

deskvox_use_package(CUDA INCDIRS ${CUDA_TOOLKIT_INCLUDE})

deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

set(RAYREND_HEADERS
    ../../vvraycaster.h
)

add_definitions(-DHAVE_CONFIG_H)
add_definitions(-DVV_ARCH_CUDA=1)

deskvox_cuda_compiles(RAYREND_CUDA_SOURCES
    ../../vvraycaster.cu
)

deskvox_add_library(rayrendcuda
    ${RAYREND_HEADERS}
    ${RAYREND_CUDA_SOURCES}
)

set_target_properties(rayrendcuda PROPERTIES LINKER_LANGUAGE CXX)

# MSVC + CUDA: link with legacy stdio library

if (MSVC_VERSION GREATER 1800 AND DESKVOX_USE_CUDA AND CUDA_FOUND)
    target_link_libraries(rayrendcuda legacy_stdio_definitions.lib)
endif()

endif(DESKVOX_USE_CUDA)
