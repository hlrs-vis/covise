# - Find microhttpd
# Find the microhttpd includes and library
#
#  MICROHTTPD_INCLUDE_DIR - Where to find microhttpd includes
#  MICROHTTPD_LIBRARIES   - List of libraries when using microhttpd
#  MICROHTTPD_FOUND       - True if microhttpd was found

IF(MICROHTTPD_INCLUDE_DIR)
  SET(MICROHTTPD_FIND_QUIETLY TRUE)
ENDIF(MICROHTTPD_INCLUDE_DIR)

FIND_PATH(MICROHTTPD_INCLUDE_DIR "microhttpd.h"
  PATHS
  $ENV{MICROHTTPD_HOME}/include
  $ENV{EXTERNLIBS}/microhttpd/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "microhttpd - Headers"
)

SET(MICROHTTPD_NAMES microhttpd)
SET(MICROHTTPD_DBG_NAMES microhttpdd)

FIND_LIBRARY(MICROHTTPD_LIBRARY NAMES ${MICROHTTPD_NAMES}
  PATHS
  $ENV{MICROHTTPD_HOME}
  $ENV{EXTERNLIBS}/microhttpd
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "microhttpd - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(MICROHTTPD_LIBRARY_DEBUG NAMES ${MICROHTTPD_DBG_NAMES}
    PATHS
    $ENV{MICROHTTPD_HOME}/lib
    $ENV{EXTERNLIBS}/microhttpd/lib
    DOC "microhttpd - Library (Debug)"
  )
  
  IF(MICROHTTPD_LIBRARY_DEBUG AND MICROHTTPD_LIBRARY)
    SET(MICROHTTPD_LIBRARIES optimized ${MICROHTTPD_LIBRARY} debug ${MICROHTTPD_LIBRARY_DEBUG})
  ENDIF(MICROHTTPD_LIBRARY_DEBUG AND MICROHTTPD_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(Microhttpd DEFAULT_MSG MICROHTTPD_LIBRARY MICROHTTPD_LIBRARY_DEBUG MICROHTTPD_INCLUDE_DIR)

  MARK_AS_ADVANCED(MICROHTTPD_LIBRARY MICROHTTPD_LIBRARY_DEBUG MICROHTTPD_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(MICROHTTPD_LIBRARIES ${MICROHTTPD_LIBRARY} ${microhttpdPLATFORM_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(Microhttpd DEFAULT_MSG MICROHTTPD_LIBRARY MICROHTTPD_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(MICROHTTPD_LIBRARY microhttpdPLATFORM_LIBRARY MICROHTTPD_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(MICROHTTPD_FOUND)
  SET(MICROHTTPD_INCLUDE_DIRS ${MICROHTTPD_INCLUDE_DIR})
ENDIF(MICROHTTPD_FOUND)
