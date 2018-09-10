include(CheckCXXSourceCompiles)

find_package(Boost)
find_package(GLEW )
find_package(Visionaray)

if(NOT VISIONARAY_FOUND)
    message(STATUS "Visionaray not found, not building ray casting plugins")
    return()
endif()

deskvox_use_package(Boost)
deskvox_use_package(GLEW)
deskvox_use_package(VISIONARAY)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=0")
endif()

# Visionaray requires modern C++ language features
# Try to compile a tiny program that includes some
# Visionaray headers and see if we can compile them
set(CMAKE_REQUIRED_INCLUDES "${VISIONARAY_INCLUDE_DIR}")
check_cxx_source_compiles(
    "
    #include <visionaray/texture/texture.h>
    #include <visionaray/aligned_vector.h>
    #include <visionaray/render_target.h>
    #include <visionaray/scheduler.h>

    int main()
    {
        return 0;
    }
    "
    CXX_CAN_COMPILE_VISIONARAY
    )

if(NOT CXX_CAN_COMPILE_VISIONARAY)
    message(STATUS "Not able to compile a simple Visionaray program, not building ray casting plugins")
    return()
endif()


add_subdirectory(avx)
add_subdirectory(avx2)
add_subdirectory(cuda)
add_subdirectory(fpu)
add_subdirectory(sse2)
add_subdirectory(sse4_1)
