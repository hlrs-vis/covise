find_path(NETCDF_INCLUDE_DIR netcdf.h)
find_path(NETCDF_C++_INCLUDE_DIR netcdf) 

find_library(NETCDF_LIBRARY NAMES netcdf) 
find_library(NETCDF_LIBRARY_DEBUG NAMES netcdfd) 
find_library(NETCDF_C++_LIBRARY NAMES  netcdf-cxx4 netcdf_c++4) 
find_library(NETCDF_C++_LIBRARY_DEBUG NAMES netcdf-cxx4d netcdf_c++4d)

if(MSVC)
    # VisualStudio needs a debug version
    find_library(NETCDF_C++_LIBRARY_DEBUG NAMES ${NETCDF_C++_DBG_NAMES}
        PATHS
        $ENV{EXTERNLIBS}/NETCDF
        PATH_SUFFIXES lib64 lib libs64 libs libs/Win32 libs/Win64
        DOC "NETCDF_C++ - Library (Debug)"
    )
    find_library(NETCDF_LIBRARY_DEBUG NAMES ${NETCDF_DBG_NAMES}
        PATHS
        $ENV{EXTERNLIBS}/NETCDF
        PATH_SUFFIXES lib64 lib libs64 libs libs/Win32 libs/Win64
        DOC "NETCDF - Library (Debug)"
    )

    find_package_handle_standard_args(NETCDF DEFAULT_MSG NETCDF_C++_LIBRARY NETCDF_C++_LIBRARY_DEBUG NETCDF_INCLUDE_DIR NETCDF_C++_INCLUDE_DIR)
    if (NETCDF_FOUND)
        set(NETCDF_LIBRARIES optimized ${NETCDF_LIBRARY} debug ${NETCDF_LIBRARY_DEBUG} optimized ${NETCDF_C++_LIBRARY} debug ${NETCDF_C++_LIBRARY_DEBUG})
        set(NETCDF_C++_LIBRARIES optimized ${NETCDF_C++_LIBRARY} debug ${NETCDF_C++_LIBRARY_DEBUG})
    endif()

    mark_as_advanced(NETCDF_C++_LIBRARY NETCDF_C++_LIBRARY_DEBUG NETCDF_LIBRARY NETCDF_LIBRARY_DEBUG NETCDF_INCLUDE_DIR NETCDF_C++_INCLUDE_DIR)
else(MSVC)
    find_package_handle_standard_args(NETCDF DEFAULT_MSG NETCDF_LIBRARY NETCDF_C++_LIBRARY NETCDF_INCLUDE_DIR NETCDF_C++_INCLUDE_DIR)
    if (NETCDF_FOUND)
        set(NETCDF_LIBRARIES ${NETCDF_LIBRARY} ${NETCDF_C++_LIBRARY})
    endif()
endif(MSVC)

if (NETCDF_FOUND)
    set(NETCDF_INCLUDE_DIRS ${NETCDF_INCLUDE_DIR} ${NETCDF_C++_INCLUDE_DIR})
endif()
