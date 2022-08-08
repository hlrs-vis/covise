find_path(OpenNURBS_INCLUDE_DIR opennurbs.h)

find_library(OpenNURBS_LIBRARY NAMES opennurbs_public.lib) 
find_library(OpenNURBS_LIBRARY_DEBUG NAMES opennurbs_publicd.lib) 

if(MSVC)
    # VisualStudio needs a debug version
    find_library(OpenNURBS_LIBRARY_DEBUG NAMES ${OpenNURBS_DBG_NAMES}
        PATHS
        $ENV{EXTERNLIBS}/OpenNURBS
        PATH_SUFFIXES lib64 lib libs64 libs libs/Win32 libs/Win64
        DOC "OpenNURBS - Library (Debug)"
    )

    find_package_handle_standard_args(OpenNURBS DEFAULT_MSG   OpenNURBS_INCLUDE_DIR )
    if (OpenNURBS_FOUND)
        set(OpenNURBS_LIBRARIES optimized ${OpenNURBS_LIBRARY} debug ${OpenNURBS_LIBRARY_DEBUG})
    endif()

    mark_as_advanced( OpenNURBS_LIBRARY OpenNURBS_LIBRARY_DEBUG OpenNURBS_INCLUDE_DIR)
else(MSVC)
    find_package_handle_standard_args(OpenNURBS DEFAULT_MSG OpenNURBS_LIBRARY  OpenNURBS_INCLUDE_DIR )
    if (OpenNURBS_FOUND)
        set(OpenNURBS_LIBRARIES ${OpenNURBS_LIBRARY})
    endif()
endif(MSVC)

if (OpenNURBS_FOUND)
    set(OpenNURBS_INCLUDE_DIRS ${OpenNURBS_INCLUDE_DIR} )
endif()
