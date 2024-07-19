# - Find PROJ
# Find the PROJ includes and library
#
#  PROJ_INCLUDE_DIR - Where to find PROJ includes
#  PROJ_LIBRARIES   - List of libraries when using PROJ
#  PROJ_FOUND       - True if PROJ was found

if(PROJ_INCLUDE_DIR)
    set(PROJ_FIND_QUIETLY TRUE)
endif(PROJ_INCLUDE_DIR)

find_path(PROJ_PREFIX "include/proj.h" DOC "PROJ - Prefix")
if(NOT PROJ_PREFIX)
    find_path(
        PROJ_PREFIX "include/proj_api.h"
        PATHS $ENV{EXTERNLIBS}/proj4
        DOC "PROJ - Prefix")
endif()

find_path(
    PROJ_INCLUDE_DIR "proj.h"
    PATHS ${PROJ_PREFIX}
    PATH_SUFFIXES include
    DOC "PROJ - Headers")
set(PROJ_API FALSE)
if(NOT PROJ_INCLUDE_DIR)
    find_path(
        PROJ_INCLUDE_DIR "proj_api.h"
        PATHS ${PROJ_PREFIX}
        PATH_SUFFIXES include
        DOC "PROJ - Headers")
    set(PROJ_API TRUE)
endif()

set(PROJ_NAMES Proj4 proj proj_4_9)
set(PROJ_DBG_NAMES Proj4D projD proj_d proj_4_9_D)

find_library(
    PROJ_LIBRARY
    NAMES ${PROJ_NAMES}
    PATHS ${PROJ_PREFIX}
    PATH_SUFFIXES lib lib64
    DOC "PROJ - Library")

include(FindPackageHandleStandardArgs)

if(MSVC)
    # VisualStudio needs a debug version
    find_library(
        PROJ_LIBRARY_DEBUG
        NAMES ${PROJ_DBG_NAMES}
        PATHS ${PROJ_PREFIX}/lib
        DOC "PROJ - Library (Debug)")

    if(PROJ_LIBRARY_DEBUG AND PROJ_LIBRARY)
        set(PROJ_LIBRARIES optimized ${PROJ_LIBRARY} debug ${PROJ_LIBRARY_DEBUG})
    endif(PROJ_LIBRARY_DEBUG AND PROJ_LIBRARY)

    find_package_handle_standard_args(PROJ DEFAULT_MSG PROJ_LIBRARY PROJ_LIBRARY_DEBUG PROJ_INCLUDE_DIR)

    mark_as_advanced(PROJ_LIBRARY PROJ_LIBRARY_DEBUG PROJ_INCLUDE_DIR PROJ_API)

else(MSVC)
    # rest of the world
    set(PROJ_LIBRARIES ${PROJ_LIBRARY})

    find_package_handle_standard_args(PROJ DEFAULT_MSG PROJ_LIBRARY PROJ_INCLUDE_DIR)

    mark_as_advanced(PROJ_LIBRARY PROJ_INCLUDE_DIR PROJ_API)

endif(MSVC)

if(PROJ_FOUND)
    set(PROJ_INCLUDE_DIRS ${PROJ_INCLUDE_DIR})
endif(PROJ_FOUND)
