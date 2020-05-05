# - Find RBDL
# Find the RBDL includes and library
#
#  RBDL_INCLUDE_DIR - Where to find RBDL includes
#  RBDL_LIBRARIES   - List of libraries when using RBDL
#  RBDL_FOUND       - True if RBDL was found

IF(RBDL_INCLUDE_DIR)
  SET(RBDL_FIND_QUIETLY TRUE)
ENDIF(RBDL_INCLUDE_DIR)

FIND_PATH(RBDL_INCLUDE_DIR "rbdl/rbdl.h"
  PATHS
  $ENV{RBDL_HOME}/include
  $ENV{EXTERNLIBS}/RBDL/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "RBDL - Headers"
)

SET(RBDL_NAMES rbdl)
SET(RBDL_DBG_NAMES rbdld)

FIND_LIBRARY(RBDL_LIBRARY NAMES ${RBDL_NAMES}
  PATHS
  $ENV{RBDL_HOME}
  $ENV{EXTERNLIBS}/RBDL
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "RBDL - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(RBDL_LIBRARY_DEBUG NAMES ${RBDL_DBG_NAMES}
    PATHS
    $ENV{RBDL_HOME}/lib
    $ENV{EXTERNLIBS}/RBDL/lib
    DOC "RBDL - Library (Debug)"
  )
  
  IF(RBDL_LIBRARY_DEBUG AND RBDL_LIBRARY)
    SET(RBDL_LIBRARIES optimized ${RBDL_LIBRARY} debug ${RBDL_LIBRARY_DEBUG})
  ENDIF(RBDL_LIBRARY_DEBUG AND RBDL_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(RBDL DEFAULT_MSG RBDL_LIBRARY RBDL_LIBRARY_DEBUG RBDL_INCLUDE_DIR)

  MARK_AS_ADVANCED(RBDL_LIBRARY RBDL_LIBRARY_DEBUG RBDL_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(RBDL_LIBRARIES ${RBDL_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(RBDL DEFAULT_MSG RBDL_LIBRARY RBDL_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(RBDL_LIBRARY RBDL_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(RBDL_FOUND)
  SET(RBDL_INCLUDE_DIRS ${RBDL_INCLUDE_DIR})
ENDIF(RBDL_FOUND)
