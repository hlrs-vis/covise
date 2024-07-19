# - Find PROJ4
# Find the PROJ4 includes and library
#
#  PROJ4_INCLUDE_DIR - Where to find PROJ4 includes
#  PROJ4_LIBRARIES   - List of libraries when using PROJ4
#  PROJ4_FOUND       - True if PROJ4 was found

IF(PROJ_INCLUDE_DIR)
  SET(PROJ_FIND_QUIETLY TRUE)
ENDIF(PROJ_INCLUDE_DIR)

FIND_PATH(PROJ_INCLUDE_DIR "proj.h"
  PATHS
  $ENV{EXTERNLIBS}/proj
  $ENV{EXTERNLIBS}/proj4
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  PATH_SUFFIXES include
  DOC "PROJ - Headers"
)
set(PROJ_API FALSE)

if(NOT PROJ_INCLUDE_DIR)
    find_path(
        PROJ_INCLUDE_DIR "proj_api.h"
        PATHS $ENV{EXTERNLIBS}/proj $ENV{EXTERNLIBS}/proj4
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local
        /usr
        /sw # Fink
        /opt/local # DarwinPorts
        /opt/csw # Blastwave
        /opt
        PATH_SUFFIXES include
        DOC "PROJ - Prefix")
        set(PROJ_API TRUE)
endif()

SET(PROJ_NAMES Proj4 proj proj_4_9)
SET(PROJ_DBG_NAMES proj_d Proj4D projD proj_4_9_D)

FIND_LIBRARY(PROJ_LIBRARY NAMES ${PROJ_NAMES}
  PATHS
  $ENV{EXTERNLIBS}/proj
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "PROJ - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(PROJ_LIBRARY_DEBUG NAMES ${PROJ_DBG_NAMES}
    PATHS
    $ENV{EXTERNLIBS}/proj/lib
    DOC "PROJ - Library (Debug)"
  )
  
  IF(PROJ_LIBRARY_DEBUG AND PROJ_LIBRARY)
    SET(PROJ_LIBRARIES optimized ${PROJ_LIBRARY} debug ${PROJ_LIBRARY_DEBUG})
  ENDIF(PROJ_LIBRARY_DEBUG AND PROJ_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(PROJ DEFAULT_MSG PROJ_LIBRARY PROJ_LIBRARY_DEBUG PROJ_INCLUDE_DIR)

  MARK_AS_ADVANCED(PROJ_LIBRARY PROJ_LIBRARY_DEBUG PROJ_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(PROJ_LIBRARIES ${PROJ_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(PROJ DEFAULT_MSG PROJ_LIBRARY PROJ_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(PROJ_LIBRARY PROJ_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(PROJ_FOUND)
  SET(PROJ_INCLUDE_DIRS ${PROJ_INCLUDE_DIR})
ENDIF(PROJ_FOUND)
