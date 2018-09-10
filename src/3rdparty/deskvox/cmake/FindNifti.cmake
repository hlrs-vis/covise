include(FindPackageHandleStandardArgs)

set(hints
  $ENV{LIB_BASE_PATH}/nifti
)

set(paths
  /usr
  /usr/local
)

find_path(Nifti_INCLUDE_DIR
  NAMES
    nifti1.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
    include/nifti
)

find_library(NiftiIo_LIBRARY
  NAMES
    niftiio
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

find_library(NiftiZnz_LIBRARY
  NAMES
    znz
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

if(NiftiZnz_LIBRARY)
  find_package(ZLIB)
endif()

find_library(NiftiIo_LIBRARY_DEBUG
  NAMES
    niftioiod
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

find_library(NiftiZnz_LIBRARY_DEBUG
  NAMES
    znzd
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

if(NiftiIo_LIBRARY_DEBUG)
  set(Nifti_LIBRARIES optimized ${NiftiIo_LIBRARY} debug ${NiftiIo_LIBRARY_DEBUG})
else()
  set(Nifti_LIBRARIES ${NiftiIo_LIBRARY})
endif()

if(NiftiZnz_LIBRARY_DEBUG)
  set(Nifti_LIBRARIES ${Nifti_LIBRARIES} optimized ${NiftiZnz_LIBRARY} debug ${NiftiZnz_LIBRARY_DEBUG})
else()
  set(Nifti_LIBRARIES ${Nifti_LIBRARIES} ${NiftiZnz_LIBRARY})
  if (ZLIB_LIBRARY)
    set(Nifti_LIBRARIES ${Nifti_LIBRARIES} ${ZLIB_LIBRARY})
  endif()
endif()

find_package_handle_standard_args(Nifti
  DEFAULT_MSG
  Nifti_INCLUDE_DIR
  NiftiIo_LIBRARY
)
