# - Find JAKA
# Find the JAKA includes and library
#
#  JAKA_INCLUDE_DIR - Where to find JAKA includes
#  JAKA_LIBRARIES   - List of libraries when using JAKA
#  JAKA_FOUND       - True if JAKA was found

IF(JAKA_INCLUDE_DIR)
  SET(JAKA_FIND_QUIETLY TRUE)
ENDIF(JAKA_INCLUDE_DIR)

FIND_PATH(JAKA_INCLUDE_DIR "JAKAZuRobot.h"
  PATHS
  $ENV{JAKA_HOME}/include
  $ENV{EXTERNLIBS}/jaka/inc_of_c++/
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "JAKA - Headers"
)

SET(JAKA_NAMES jakaAPI)
SET(JAKA_DBG_NAMES jakaAPI)

FIND_LIBRARY(JAKA_LIBRARY NAMES ${JAKA_NAMES}
  PATHS
  $ENV{JAKA_HOME}
  $ENV{EXTERNLIBS}/jaka/x64
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "JAKA - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(JAKA_LIBRARY_DEBUG NAMES ${JAKA_DBG_NAMES}
    PATHS
  $ENV{EXTERNLIBS}/jaka/x64
    DOC "JAKA - Library (Debug)"
  )
  
  IF(JAKA_LIBRARY_DEBUG AND JAKA_LIBRARY)
    SET(JAKA_LIBRARIES optimized ${JAKA_LIBRARY} debug ${JAKA_LIBRARY_DEBUG})
  ENDIF(JAKA_LIBRARY_DEBUG AND JAKA_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(JAKA DEFAULT_MSG JAKA_LIBRARY JAKA_LIBRARY_DEBUG JAKA_INCLUDE_DIR)

  MARK_AS_ADVANCED(JAKA_LIBRARY JAKA_LIBRARY_DEBUG JAKA_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(JAKA_LIBRARIES ${JAKA_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(JAKA DEFAULT_MSG JAKA_LIBRARY JAKA_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(JAKA_LIBRARY JAKA_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(JAKA_FOUND)
  SET(JAKA_INCLUDE_DIRS ${JAKA_INCLUDE_DIR})
ENDIF(JAKA_FOUND)
