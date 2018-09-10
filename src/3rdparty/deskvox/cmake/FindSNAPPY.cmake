include(FindPackageHandleStandardArgs)

set(hints
  $ENV{EXTERNLIBS}/snappy
  $ENV{LIB_BASE_PATH}/
  $ENV{LIB_BASE_PATH}/snappy
)

set(paths
  /usr
  /usr/local
)

find_path(SNAPPY_INCLUDE_DIR
  NAMES
    snappy.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
    include/snappy
)

find_library(SNAPPY_LIBRARY
  NAMES
    snappy
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

find_library(SNAPPY_LIBRARY_DEBUG
  NAMES
    snappyd
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

if(SNAPPY_LIBRARY_DEBUG)
  set(SNAPPY_LIBRARIES optimized ${SNAPPY_LIBRARY} debug ${SNAPPY_LIBRARY_DEBUG})
else()
  set(SNAPPY_LIBRARIES ${SNAPPY_LIBRARY})
endif()

find_package_handle_standard_args(SNAPPY DEFAULT_MSG
  SNAPPY_INCLUDE_DIR
  SNAPPY_LIBRARY
)
