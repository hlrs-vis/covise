# - Find PROJ4
# Find the PROJ4 includes and library
#
#  PROJ4_INCLUDE_DIR - Where to find PROJ4 includes
#  PROJ4_LIBRARIES   - List of libraries when using PROJ4
#  PROJ4_FOUND       - True if PROJ4 was found

IF(PROJ4_INCLUDE_DIR)
  SET(PROJ4_FIND_QUIETLY TRUE)
ENDIF(PROJ4_INCLUDE_DIR)

FIND_PATH(PROJ4_INCLUDE_DIR "proj_api.h"
  PATHS
  $ENV{EXTERNLIBS}/proj4/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "PROJ4 - Headers"
)

SET(PROJ4_NAMES Proj4 proj proj_4_9)
SET(PROJ4_DBG_NAMES Proj4D projD proj_4_9_D)

FIND_LIBRARY(PROJ4_LIBRARY NAMES ${PROJ4_NAMES}
  PATHS
  $ENV{EXTERNLIBS}/proj4
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "PROJ4 - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(PROJ4_LIBRARY_DEBUG NAMES ${PROJ4_DBG_NAMES}
    PATHS
    $ENV{EXTERNLIBS}/proj4/lib
    DOC "PROJ4 - Library (Debug)"
  )
  
  IF(PROJ4_LIBRARY_DEBUG AND PROJ4_LIBRARY)
    SET(PROJ4_LIBRARIES optimized ${PROJ4_LIBRARY} debug ${PROJ4_LIBRARY_DEBUG})
  ENDIF(PROJ4_LIBRARY_DEBUG AND PROJ4_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(PROJ4 DEFAULT_MSG PROJ4_LIBRARY PROJ4_LIBRARY_DEBUG PROJ4_INCLUDE_DIR)

  MARK_AS_ADVANCED(PROJ4_LIBRARY PROJ4_LIBRARY_DEBUG PROJ4_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(PROJ4_LIBRARIES ${PROJ4_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(PROJ4 DEFAULT_MSG PROJ4_LIBRARY PROJ4_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(PROJ4_LIBRARY PROJ4_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(PROJ4_FOUND)
  SET(PROJ4_INCLUDE_DIRS ${PROJ4_INCLUDE_DIR})
ENDIF(PROJ4_FOUND)
