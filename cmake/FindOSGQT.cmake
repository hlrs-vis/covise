# - Find osgQt
# Find the alut includes and library
#
#  OSGQT_INCLUDE_DIR - Where to find OSGQT includes
#  OSGQT_LIBRARIES   - List of libraries when using OSGQT
#  OSGQT_FOUND       - True if OSGQT was found

IF(OSGQT_INCLUDE_DIR)
  SET(OSGQT_FIND_QUIETLY TRUE)
ENDIF(OSGQT_INCLUDE_DIR)

FIND_PATH(OSGQT_INCLUDE_DIR "osgQt/QWidgetImage"
  PATHS
  $ENV{OSGQT_HOME}/include
  $ENV{EXTERNLIBS}/OpenSceneGraph/include
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  DOC "OSGQT - Headers"
)

SET(OSGQT_NAMES osgQt5 osgQt)
SET(OSGQT_DBG_NAMES osgQt5d  osgQtd)

FIND_LIBRARY(OSGQT_LIBRARY NAMES ${OSGQT_NAMES}
  PATHS
  $ENV{OSGQT_HOME}
  $ENV{EXTERNLIBS}/OpenSceneGraph
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "OSGQT - Library"
)
INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OSGQT_LIBRARY_DEBUG NAMES ${OSGQT_DBG_NAMES}
    PATHS
    $ENV{OSGQT_HOME}/lib
    $ENV{EXTERNLIBS}/OpenSceneGraph/lib
    DOC "OSGQT - Library (Debug)"
  )
  IF(OSGQT_LIBRARY_DEBUG AND OSGQT_LIBRARY)
    SET(OSGQT_LIBRARIES optimized ${OSGQT_LIBRARY} debug ${OSGQT_LIBRARY_DEBUG} )
  ENDIF(OSGQT_LIBRARY_DEBUG AND OSGQT_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSGQT DEFAULT_MSG OSGQT_LIBRARY OSGQT_LIBRARY_DEBUG OSGQT_INCLUDE_DIR)

  MARK_AS_ADVANCED(OSGQT_LIBRARY OSGQT_LIBRARY_DEBUG OSGQT_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OSGQT_LIBRARIES ${OSGQT_LIBRARY} )

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSGQT DEFAULT_MSG OSGQT_LIBRARY OSGQT_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OSGQT_LIBRARY OSGQT_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OSGQT_FOUND)
  SET(OSGQT_INCLUDE_DIRS ${OSGQT_INCLUDE_DIR})
ENDIF(OSGQT_FOUND)
