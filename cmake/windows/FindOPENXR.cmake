# - Find OPENXR
# Find the OPENXR includes and library
#
#  OPENXR_INCLUDE_DIR - Where to find OPENXR includes
#  OPENXR_LIBRARIES   - List of libraries when using OPENXR
#  OPENXR_FOUND       - True if OPENXR was found

IF(OPENXR_INCLUDE_DIR)
  SET(OPENXR_FIND_QUIETLY TRUE)
ENDIF(OPENXR_INCLUDE_DIR)

FIND_PATH(OPENXR_INCLUDE_DIR "openxr/openxr.h"
  PATHS
  $ENV{OPENXR_HOME}/include
  $ENV{EXTERNLIBS}/OpenXR/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "OPENXR - Headers"
  PATH_SUFFIXES OPENXR
)

SET(OPENXR_NAMES openxr_loader openxr_loader.lib)
SET(OPENXR_DBG_NAMES openxr_loaderd openxr_loaderd.lib)

FIND_LIBRARY(OPENXR_LIBRARY NAMES ${OPENXR_NAMES}
  PATHS
  $ENV{OPENXR_HOME}
  $ENV{EXTERNLIBS}/OpenXR
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 lib/msvc_x64
  DOC "OPENXR - Library"
)
INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OPENXR_LIBRARY_DEBUG NAMES ${OPENXR_DBG_NAMES}
    PATHS
    $ENV{OPENXR_HOME}/lib
    $ENV{EXTERNLIBS}/OPENXR
    PATH_SUFFIXES bin lib lib64 lib/msvc_x64
    DOC "OPENXR - Library (Debug)"
  )
  
  
  IF(OPENXR_LIBRARY_DEBUG AND OPENXR_LIBRARY)
    SET(OPENXR_LIBRARIES optimized ${OPENXR_LIBRARY} debug ${OPENXR_LIBRARY_DEBUG})
  ENDIF(OPENXR_LIBRARY_DEBUG AND OPENXR_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENXR DEFAULT_MSG OPENXR_LIBRARY OPENXR_LIBRARY_DEBUG OPENXR_INCLUDE_DIR)

  MARK_AS_ADVANCED(OPENXR_LIBRARY OPENXR_LIBRARY_DEBUG   OPENXR_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OPENXR_LIBRARIES ${OPENXR_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENXR DEFAULT_MSG OPENXR_LIBRARY OPENXR_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OPENXR_LIBRARY OPENXR_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OPENXR_FOUND)
  SET(OPENXR_INCLUDE_DIRS ${OPENXR_INCLUDE_DIR})
ENDIF(OPENXR_FOUND)
