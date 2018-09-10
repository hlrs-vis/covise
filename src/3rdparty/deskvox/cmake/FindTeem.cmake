include(FindPackageHandleStandardArgs)

set(hints
  $ENV{LIB_BASE_PATH}/teem
)

set(paths
  /usr
  /usr/local
)

find_path(TEEM_INCLUDE_DIR
  NAMES
    teem/nrrd.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
    include/teem
)

find_library(TEEM_LIBRARY
  NAMES
    teem
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

set(FOX_LIBRARIES ${FOX_LIBRARY})

find_package_handle_standard_args(Teem DEFAULT_MSG
  TEEM_INCLUDE_DIR
  TEEM_LIBRARY
)
