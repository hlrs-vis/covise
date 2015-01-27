# - Find osgCal
# Find the osgCal includes and library
#
#  OSGCAL_INCLUDE_DIR - Where to find osgCal includes
#  OSGCAL_LIBRARIES   - List of libraries when using osgCal
#  OSGCAL_FOUND       - True if osgCal was found

IF(OSGCAL_INCLUDE_DIR)
  SET(OSGCAL_FIND_QUIETLY TRUE)
ENDIF(OSGCAL_INCLUDE_DIR)

FIND_PATH(OSGCAL_INCLUDE_DIR "osgCal/CoreMesh"
  PATHS
  $ENV{OSGCAL_HOME}/include
  $ENV{EXTERNLIBS}/osgCal/include
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "osgCal - Headers"
)

SET(OSGCAL_NAMES osgCal)
SET(OSGCAL_DBG_NAMES osgCald)

FIND_LIBRARY(OSGCAL_LIBRARY NAMES ${OSGCAL_NAMES}
  PATHS
  $ENV{OSGCAL_HOME}
  $ENV{EXTERNLIBS}/osgCal
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "osgCal - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OSGCAL_LIBRARY_DEBUG NAMES ${OSGCAL_DBG_NAMES}
    PATHS
    $ENV{OSGCAL_HOME}
    $ENV{EXTERNLIBS}/osgCal
    PATH_SUFFIXES lib lib64
    DOC "osgCal - Library (Debug)"
  )
  
  IF(OSGCAL_LIBRARY_DEBUG AND OSGCAL_LIBRARY)
    SET(OSGCAL_LIBRARIES optimized ${OSGCAL_LIBRARY} debug ${OSGCAL_LIBRARY_DEBUG})
  ENDIF(OSGCAL_LIBRARY_DEBUG AND OSGCAL_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSGCAL DEFAULT_MSG OSGCAL_LIBRARY OSGCAL_LIBRARY_DEBUG OSGCAL_INCLUDE_DIR)

  MARK_AS_ADVANCED(OSGCAL_LIBRARY OSGCAL_LIBRARY_DEBUG OSGCAL_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OSGCAL_LIBRARIES ${OSGCAL_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSGCAL DEFAULT_MSG OSGCAL_LIBRARY OSGCAL_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OSGCAL_LIBRARY OSGCAL_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OSGCAL_FOUND)
  SET(OSGCAL_INCLUDE_DIRS ${OSGCAL_INCLUDE_DIR})
ENDIF(OSGCAL_FOUND)
