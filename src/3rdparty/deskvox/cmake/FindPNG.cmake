include(FindPackageHandleStandardArgs)

set(hints
  $ENV{EXTERNLIBS}/libpng
  $ENV{LIB_BASE_PATH}/
  $ENV{LIB_BASE_PATH}/libpng/
)

set(paths
  /usr
  /usr/local
)

find_path(PNG_INCLUDE_DIR
  NAMES
    png.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
    include/libpng
    include/libpng16
)

find_library(PNG_LIBRARY
  NAMES
    png
    libpng
    libpng16
    libpng_static
    libpng16_static
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
    lib/x86_64-linux-gnu
)

find_library(PNG_LIBRARY_DEBUG
  NAMES
    pngd
    libpngd
    libpng16d
    libpng_staticd
    libpng16_staticd
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
    lib/x86_64-linux-gnu
)

if(PNG_LIBRARY_DEBUG)
  set(PNG_LIBRARIES optimized ${PNG_LIBRARY} debug ${PNG_LIBRARY_DEBUG})
else()
  set(PNG_LIBRARIES ${PNG_LIBRARY})
endif()

find_package_handle_standard_args(PNG DEFAULT_MSG
  PNG_INCLUDE_DIR
  PNG_LIBRARY
)
