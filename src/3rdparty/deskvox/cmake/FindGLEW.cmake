include(FindPackageHandleStandardArgs)

set(hints
  $ENV{EXTERNLIBS}/glew
  $ENV{LIB_BASE_PATH}/
  $ENV{LIB_BASE_PATH}/glew/
)

set(paths
  /usr
  /usr/local
)

find_path(GLEW_INCLUDE_DIR
  NAMES
    GL/glew.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
)

find_library(GLEW_LIBRARY
  NAMES
    GLEW
    glew
    glew-s
    glew32
    glew32-s
    glew32_static
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

find_library(GLEW_LIBRARY_DEBUG
  NAMES
    GLEWd
    glewd
    glewd-s
    glew32d
    glew32d-s
    glew32d_static
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

if(GLEW_LIBRARY_DEBUG)
  set(GLEW_LIBRARIES optimized ${GLEW_LIBRARY} debug ${GLEW_LIBRARY_DEBUG})
else()
  set(GLEW_LIBRARIES ${GLEW_LIBRARY})
endif()

find_package_handle_standard_args(GLEW DEFAULT_MSG
  GLEW_INCLUDE_DIR
  GLEW_LIBRARY
)
