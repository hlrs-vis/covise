# - Find Faro
# Find the Faro includes and library
#
#  FARO_INCLUDE_DIR - Where to find FARO includes
#  FARO_LIBRARIES   - List of libraries when using Faro
#  FARO_FOUND       - True if Faro was found

IF(FARO_INCLUDE_DIR)
  SET(FARO_FIND_QUIETLY TRUE)
ENDIF(FARO_INCLUDE_DIR)

FIND_PATH(FARO_INCLUDE_DIR "FaroScannerAPI.h"
  PATHS
  $ENV{FARO_HOME}/Inc
  $ENV{EXTERNLIBS}/Faro/Inc
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  DOC "Faro - Headers"
)

SET(FARO_NAMES FaroLaserScannerAPI)
SET(FARO_DBG_NAMES FaroLaserScannerAPId)

FIND_LIBRARY(FARO_LIBRARY NAMES ${FARO_NAMES}
  PATHS
  $ENV{FARO_HOME}
  $ENV{EXTERNLIBS}/Faro
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "Faro - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(FARO_LIBRARY_DEBUG NAMES ${FARO_DBG_NAMES}
    PATHS
    $ENV{FARO_HOME}/lib
    $ENV{EXTERNLIBS}/Faro/lib
    DOC "Faro - Library (Debug)"
  )
  
  IF(FARO_LIBRARY_DEBUG AND FARO_LIBRARY)
    SET(FARO_LIBRARIES optimized ${FARO_LIBRARY} debug ${FARO_LIBRARY_DEBUG})
  ENDIF(FARO_LIBRARY_DEBUG AND FARO_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FARO DEFAULT_MSG FARO_LIBRARY FARO_LIBRARY_DEBUG FARO_INCLUDE_DIR)

  MARK_AS_ADVANCED(FARO_LIBRARY FARO_LIBRARY_DEBUG FARO_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(FARO_LIBRARIES ${FARO_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FARO DEFAULT_MSG FARO_LIBRARY FARO_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(FARO_LIBRARY FARO_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(FARO_FOUND)
  SET(FARO_INCLUDE_DIRS ${FARO_INCLUDE_DIR})
ENDIF(FARO_FOUND)
