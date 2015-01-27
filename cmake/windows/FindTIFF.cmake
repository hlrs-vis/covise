# - Try to find tiff-library
# Once done this will define
#
#  TIFF_INCLUDE_DIR    - where to find tiff.h, etc.
#  TIFF_INCLUDE_DIRS   - same as above (uncached version)
#  TIFF_LIBRARIES      - list of libraries when using tiff.
#  TIFF_FOUND          - True if tiff was found.

IF(TIFF_INCLUDE_DIR)
  SET(TIFF_FIND_QUIETLY TRUE)
ENDIF(TIFF_INCLUDE_DIR)

FIND_PATH(TIFF_INCLUDE_DIR tiff.h
   PATHS
   $ENV{TIFF_HOME}/include
   $ENV{EXTERNLIBS}/tiff/include
   DOC "tiff - Headers"
   NO_DEFAULT_PATH
)
FIND_PATH(TIFF_INCLUDE_DIR tiff.h DOC "tiff - Headers")
MARK_AS_ADVANCED(TIFF_INCLUDE_DIR)

IF (MSVC)
    # check whether this is a /MT(d) build
    STRING(REGEX MATCH "[mM][tT][dD]" MTD_COMPILE_OPTION ${CMAKE_C_FLAGS_DEBUG})
    IF (MTD_COMPILE_OPTION)
      # MESSAGE("Using static MS-Runtime !!!")
      FIND_LIBRARY(TIFF_LIBRARY_DEBUG NAMES tiffd_mt libtiffd_mt
        PATHS
        $ENV{TIFF_HOME}/lib
        $ENV{EXTERNLIBS}/tiff/lib
      )
      FIND_LIBRARY(TIFF_LIBRARY_RELEASE NAMES tiff_mt libtiff_mt
        PATHS
        $ENV{TIFF_HOME}/lib
        $ENV{EXTERNLIBS}/tiff/lib
      )
    ELSE (MTD_COMPILE_OPTION)
      FIND_LIBRARY(TIFF_LIBRARY_DEBUG NAMES tiffd tiff-staticd libtiffd libtiff15_staticd tiffd_i tiff15d_i
        PATHS
        $ENV{TIFF_HOME}/lib
        $ENV{EXTERNLIBS}/tiff/lib
      )
      FIND_LIBRARY(TIFF_LIBRARY_RELEASE NAMES tiff tiff-static libtiff libtiff15_static tiff_i tiff15_i
        PATHS
        $ENV{TIFF_HOME}/lib
        $ENV{EXTERNLIBS}/tiff/lib
      )
    ENDIF (MTD_COMPILE_OPTION)

    IF(MSVC_IDE)
      IF (TIFF_LIBRARY_DEBUG AND TIFF_LIBRARY_RELEASE)
         SET(TIFF_LIBRARIES optimized ${TIFF_LIBRARY_RELEASE} debug ${TIFF_LIBRARY_DEBUG})
      ELSE (TIFF_LIBRARY_DEBUG AND TIFF_LIBRARY_RELEASE)
         SET(TIFF_LIBRARIES NOTFOUND)
         MESSAGE(STATUS "Could not find the debug AND release version of tiff")
      ENDIF (TIFF_LIBRARY_DEBUG AND TIFF_LIBRARY_RELEASE)
    ELSE(MSVC_IDE)
      STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_TOLOWER)
      IF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(TIFF_LIBRARIES ${TIFF_LIBRARY_DEBUG})
      ELSE(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(TIFF_LIBRARIES ${TIFF_LIBRARY_RELEASE})
      ENDIF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
    ENDIF(MSVC_IDE)
    MARK_AS_ADVANCED(TIFF_LIBRARY_DEBUG TIFF_LIBRARY_RELEASE)

ELSE (MSVC)
  
  SET(TIFF_NAMES ${TIFF_NAMES} tiff libtiff tiff3 libtiff3 tiffd libtiffd tiff3d libtiff3d)
  FIND_LIBRARY(TIFF_LIBRARY NAMES ${TIFF_NAMES}
    PATHS
    $ENV{TIFF_HOME}/lib
    $ENV{EXTERNLIBS}/tiff/lib
    NO_DEFAULT_PATH
  )
  FIND_LIBRARY(TIFF_LIBRARY NAMES ${TIFF_NAMES})

  IF (TIFF_LIBRARY)
    SET(TIFF_LIBRARIES ${TIFF_LIBRARY})
  ELSE (TIFF_LIBRARY)
    SET(TIFF_LIBRARIES NOTFOUND)
    MESSAGE(STATUS "Could not find tiff-library")    
  ENDIF (TIFF_LIBRARY)
  MARK_AS_ADVANCED(TIFF_LIBRARY)
  
ENDIF (MSVC)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(TIFF DEFAULT_MSG TIFF_LIBRARY_RELEASE TIFF_LIBRARY_DEBUG TIFF_INCLUDE_DIR)
  MARK_AS_ADVANCED(TIFF_LIBRARY_RELEASE TIFF_LIBRARY_DEBUG)
ELSE(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(TIFF DEFAULT_MSG TIFF_LIBRARY TIFF_INCLUDE_DIR)
  MARK_AS_ADVANCED(TIFF_LIBRARY)
ENDIF(MSVC)
