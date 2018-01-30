# - Find SISL
# Find the SISL includes and library
#
#  SISL_INCLUDE_DIR - Where to find SISL includes
#  SISL_LIBRARIES   - List of libraries when using SISL
#  SISL_FOUND       - True if SISL was found

IF(SISL_INCLUDE_DIR)
  SET(SISL_FIND_QUIETLY TRUE)
ENDIF(SISL_INCLUDE_DIR)

FIND_PATH(SISL_INCLUDE_DIR "sisl.h"
  PATHS
  $ENV{SISL_HOME}/include
  $ENV{EXTERNLIBS}/SISL/usr/local/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "SISL - Headers"
)

SET(SISL_NAMES sisl)
SET(SISL_DBG_NAMES sisl)

FIND_LIBRARY(SISL_LIBRARY NAMES ${SISL_NAMES}
  PATHS
  $ENV{SISL_HOME}
  $ENV{EXTERNLIBS}/SISL/usr/local/lib
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "SISL - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(SISL_LIBRARY_DEBUG NAMES ${SISL_DBG_NAMES}
    PATHS
    $ENV{SISL_HOME}/lib
    $ENV{EXTERNLIBS}/SISL/lib
    DOC "SISL - Library (Debug)"
  )
  
  IF(SISL_LIBRARY_DEBUG AND SISL_LIBRARY)
    SET(SISL_LIBRARIES optimized ${SISL_LIBRARY} debug ${SISL_LIBRARY_DEBUG})
  ENDIF(SISL_LIBRARY_DEBUG AND SISL_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(SISL DEFAULT_MSG SISL_LIBRARY SISL_LIBRARY_DEBUG SISL_INCLUDE_DIR)

  MARK_AS_ADVANCED(SISL_LIBRARY SISL_LIBRARY_DEBUG SISL_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(SISL_LIBRARIES ${SISL_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(SISL DEFAULT_MSG SISL_LIBRARY SISL_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(SISL_LIBRARY SISL_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(SISL_FOUND)
  SET(SISL_INCLUDE_DIRS ${SISL_INCLUDE_DIR})
ENDIF(SISL_FOUND)
