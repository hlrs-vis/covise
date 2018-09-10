include(FindPackageHandleStandardArgs)

set(hints
  $ENV{LIB_BASE_PATH}/
  $ENV{LIB_BASE_PATH}/pthread
  $ENV{LIB_BASE_PATH}/pthreads
)

set(paths
  /usr
  /usr/local
)

find_path(Pthreads_INCLUDE_DIR
  NAMES
    pthread.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
    include/pthread
    include/pthreads
)

if(NOT Pthreads_EXCEPTION_SCHEME)
  set(Pthreads_EXCEPTION_SCHEME "C")
else()
  if(NOT Pthreads_EXCEPTION_SCHEME STREQUAL "C" AND
     NOT Pthreads_EXCEPTION_SCHEME STREQUAL "CE" AND
     NOT Pthreads_EXCEPTION_SCHEME STREQUAL "SE")
    message(FATAL_ERROR "Only C, CE, and SE exception modes are allowed")
  endif()
  if(NOT MSVC AND Pthreads_EXCEPTION_SCHEME STREQUAL "SE")
    message(FATAL_ERROR "Structured Exception Handling is only allowed for MSVC")
  endif()
endif()

if(MSVC)
  set(Pthreads_PREFIX "V")
else()
  set(Pthreads_PREFIX "G")
endif()

find_library(Pthreads_LIBRARY
  NAMES
    pthread${Pthreads_PREFIX}${Pthreads_EXCEPTION_SCHEME}2
    pthread
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

find_library(Pthreads_LIBRARY_DEBUG
  NAMES
    pthread${Pthreads_PREFIX}${Pthreads_EXCEPTION_SCHEME}2d
    pthreadd
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

if(Pthreads_LIBRARY_DEBUG)
  set(Pthreads_LIBRARIES optimized ${Pthreads_LIBRARY} debug ${Pthreads_LIBRARY_DEBUG})
else()
  set(Pthreads_LIBRARIES ${Pthreads_LIBRARY})
endif()

if(Pthreads_EXCEPTION_SCHEME STREQUAL "C")
  set(Pthreads_PACKAGE_DEFINITIONS "-D__CLEANUP_C")
elseif(Pthreads_EXCEPTION_SCHEME STREQUAL "CE")
  set(Pthreads_PACKAGE_DEFINITIONS "-D__CLEANUP_CXX")
elseif(Pthreads_EXCEPTION_SCHEME STREQUAL "SE")
  set(Pthreads_PACKAGE_DEFINITIONS "-D__CLEANUP_SEH")
endif()

# TODO:
# Command line option -lpthread gets defined twice...
if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if(MINGW)
    set(Pthreads_PACKAGE_CXX_FLAGS "-mthreads")
  else()
    set(Pthreads_PACKAGE_CXX_FLAGS "-pthread")
  endif()
endif()

find_package_handle_standard_args(Pthreads DEFAULT_MSG
  Pthreads_INCLUDE_DIR
  Pthreads_LIBRARY
)
