# - Try to find png-library
# Once done this will define
#
#  PNG_INCLUDE_DIR    - where to find png.h, etc.
#  PNG_INCLUDE_DIRS   - same as above (uncached version)
#  PNG_LIBRARIES      - list of libraries when using png.
#  PNG_DEFINITIONS    - use ADD_DEFINITIONS(${PNG_DEFINITIONS})
#  PNG_FOUND          - True if png was found.

IF(PNG_INCLUDE_DIR)
  SET(PNG_FIND_QUIETLY TRUE)
ENDIF(PNG_INCLUDE_DIR)

FIND_PATH(PNG_INCLUDE_DIR png.h
   PATHS
   $ENV{PNG_HOME}/include
   $ENV{EXTERNLIBS}/png/include
   /usr/local/include/libpng
   DOC "png - Headers"
   NO_DEFAULT_PATH
)
FIND_PATH(PNG_INCLUDE_DIR png.h DOC "png - Headers")

MARK_AS_ADVANCED(PNG_INCLUDE_DIR)

IF (MSVC)
    # check whether this is a /MT(d) build
    STRING(REGEX MATCH "[mM][tT][dD]" MTD_COMPILE_OPTION ${CMAKE_C_FLAGS_DEBUG})
    IF (MTD_COMPILE_OPTION)
      # MESSAGE("Using static MS-Runtime !!!")
      FIND_LIBRARY(PNG_LIBRARY_DEBUG NAMES pngd_mt
        PATHS
        $ENV{PNG_HOME}/lib
        $ENV{EXTERNLIBS}/png/lib
      )
      FIND_LIBRARY(PNG_LIBRARY_RELEASE NAMES png_mt
        PATHS
        $ENV{PNG_HOME}/lib
        $ENV{EXTERNLIBS}/png/lib
      )
    ELSE (MTD_COMPILE_OPTION)
      FIND_LIBRARY(PNG_LIBRARY_DEBUG NAMES pngd png-staticd libpngd libpng15_staticd libpng16d libpng16staticd pngd_i png15d_i
        PATHS
        $ENV{PNG_HOME}/lib
        $ENV{EXTERNLIBS}/png/lib
      )
      FIND_LIBRARY(PNG_LIBRARY_RELEASE NAMES png png-static libpng libpng15_static libpng16 libpng16static png_i png15_i
        PATHS
        $ENV{PNG_HOME}/lib
        $ENV{EXTERNLIBS}/png/lib
      )
    ENDIF (MTD_COMPILE_OPTION)

    IF(MSVC_IDE)
      IF (PNG_LIBRARY_DEBUG AND PNG_LIBRARY_RELEASE)
         SET(PNG_LIBRARIES optimized ${PNG_LIBRARY_RELEASE} debug ${PNG_LIBRARY_DEBUG})
      ELSE (PNG_LIBRARY_DEBUG AND PNG_LIBRARY_RELEASE)
         SET(PNG_LIBRARIES NOTFOUND)
         MESSAGE(STATUS "Could not find the debug AND release version of png")
      ENDIF (PNG_LIBRARY_DEBUG AND PNG_LIBRARY_RELEASE)
    ELSE(MSVC_IDE)
      STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_TOLOWER)
      IF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(PNG_LIBRARIES ${PNG_LIBRARY_DEBUG})
      ELSE(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(PNG_LIBRARIES ${PNG_LIBRARY_RELEASE})
      ENDIF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
    ENDIF(MSVC_IDE)
    MARK_AS_ADVANCED(PNG_LIBRARY_DEBUG PNG_LIBRARY_RELEASE)
    
    # try to figure out whether we are using an "import-lib"
    STRING(REGEX MATCH "_[iI].[lL][iI][bB]$" PNG_USING_IMPORT_LIB "${PNG_LIBRARY_RELEASE}")
    IF(PNG_USING_IMPORT_LIB)
      SET(PNG_DEFINITIONS -DPNG_USE_DLL)
    ELSE(PNG_USING_IMPORT_LIB)
      SET(PNG_DEFINITIONS -DPNG_STATIC)
    ENDIF(PNG_USING_IMPORT_LIB)
    
    # MESSAGE("PNG_DEFINITIONS = ${PNG_DEFINITIONS}")

ELSE (MSVC)
  
  SET(PNG_NAMES ${PNG_NAMES} png libpng png15 libpng15 png15d libpng15d png14 libpng14 png14d libpng14d png12 libpng12 png12d libpng12d)
  FIND_LIBRARY(PNG_LIBRARY NAMES ${PNG_NAMES}
    PATHS
    $ENV{PNG_HOME}/lib
    $ENV{EXTERNLIBS}/png/lib
    NO_DEFAULT_PATH
  )
  FIND_LIBRARY(PNG_LIBRARY NAMES ${PNG_NAMES})

  IF (PNG_LIBRARY)
    SET(PNG_LIBRARIES ${PNG_LIBRARY})
  ELSE (PNG_LIBRARY)
    SET(PNG_LIBRARIES NOTFOUND)
    MESSAGE(STATUS "Could not find png-library")    
  ENDIF (PNG_LIBRARY)
  MARK_AS_ADVANCED(PNG_LIBRARY)
  
ENDIF (MSVC)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(PNG DEFAULT_MSG PNG_LIBRARY_RELEASE PNG_LIBRARY_DEBUG PNG_INCLUDE_DIR)
  MARK_AS_ADVANCED(PNG_LIBRARY_RELEASE PNG_LIBRARY_DEBUG)
ELSE(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(PNG DEFAULT_MSG PNG_LIBRARY PNG_INCLUDE_DIR)
  MARK_AS_ADVANCED(PNG_LIBRARY)
ENDIF(MSVC)
