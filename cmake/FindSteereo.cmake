# - Try to find steereo-library
# Once done this will define
#
#  STEEREO_INCLUDE_DIR    - where to find steering headers
#  STEEREO_INCLUDE_DIRS   - same as above (uncached version)
#  STEEREO_LIBRARIES      - list of libraries when using steereo
#  STEEREO_FOUND          - True if steereo was found.

IF(STEEREO_INCLUDE_DIR)
  SET(STEEREO_FIND_QUIETLY TRUE)
ENDIF(STEEREO_INCLUDE_DIR)

FIND_PATH(STEEREO_INCLUDE_DIR steereo/steereoCommand.h
   PATHS
   $ENV{STEEREO_HOME}/include
   $ENV{EXTERNLIBS}/steereo/include
   DOC "steereo - Headers"
   NO_DEFAULT_PATH
)
FIND_PATH(STEEREO_INCLUDE_DIR steereo/steereoCommand.h DOC "steereo - Headers")

MARK_AS_ADVANCED(STEEREO_INCLUDE_DIR)

FIND_LIBRARY(STEEREO_LIBRARY NAMES Steereo  STEEREO
  PATHS
  $ENV{STEEREO_HOME}/lib
   $ENV{EXTERNLIBS}/steereo/lib
  NO_DEFAULT_PATH
)
FIND_LIBRARY(STEEREO_LIBRARY NAMES Steereo  STEEREO)

IF (STEEREO_LIBRARY)
  SET(STEEREO_LIBRARIES ${STEEREO_LIBRARY})
ELSE (STEEREO_LIBRARY)
  SET(STEEREO_LIBRARIES NOTFOUND)
ENDIF (STEEREO_LIBRARY)

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Steereo DEFAULT_MSG STEEREO_LIBRARY STEEREO_INCLUDE_DIR)
MARK_AS_ADVANCED(STEEREO_LIBRARY)
