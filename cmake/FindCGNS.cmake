# - Find cgns
# Find the cgns includes and library
#
#  CGNS_INCLUDE_DIR - Where to find cgns includes
#  CGNS_LIBRARIES   - List of libraries when using cgns
#  CGNS_FOUND       - True if cgns was found

IF(CGNS_INCLUDE_DIR)
  SET(CGNS_FIND_QUIETLY TRUE)
ENDIF(CGNS_INCLUDE_DIR)

FIND_PATH(CGNS_INCLUDE_DIR "cgnslib.h"
  PATHS
  $ENV{CGNS_HOME}/include
  $ENV{EXTERNLIBS}/cgns/include
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  DOC "cgns - Headers"
)

SET(CGNS_NAMES cgns)
SET(CGNS_DBG_NAMES cgnsd)

FIND_LIBRARY(CGNS_LIBRARY NAMES ${CGNS_NAMES}
  PATHS
  $ENV{CGNS_HOME}
  $ENV{EXTERNLIBS}/cgns
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "cgns - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(CGNS_LIBRARY_DEBUG NAMES ${CGNS_DBG_NAMES}
    PATHS
    $ENV{CGNS_HOME}
    $ENV{EXTERNLIBS}/cgns
    PATH_SUFFIXES lib lib64
    DOC "cgns - Library (Debug)"
  )
  
  IF(CGNS_LIBRARY_DEBUG AND CGNS_LIBRARY)
    SET(CGNS_LIBRARIES optimized ${CGNS_LIBRARY} debug ${CGNS_LIBRARY_DEBUG})
  ENDIF(CGNS_LIBRARY_DEBUG AND CGNS_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(CGNS DEFAULT_MSG CGNS_LIBRARY CGNS_LIBRARY_DEBUG CGNS_INCLUDE_DIR)

  MARK_AS_ADVANCED(CGNS_LIBRARY CGNS_LIBRARY_DEBUG CGNS_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(CGNS_LIBRARIES ${CGNS_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(CGNS DEFAULT_MSG CGNS_LIBRARY CGNS_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(CGNS_LIBRARY CGNS_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(CGNS_FOUND)
  SET(CGNS_INCLUDE_DIRS ${CGNS_INCLUDE_DIR})
ENDIF(CGNS_FOUND)