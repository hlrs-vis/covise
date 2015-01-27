# - Find OSC
# Find the OSC includes and library
#
#  OSC_INCLUDE_DIR - Where to find OSC includes
#  OSC_LIBRARIES   - List of libraries when using OSC
#  OSC_FOUND       - True if OSC was found

IF(OSC_INCLUDE_DIR)
  SET(OSC_FIND_QUIETLY TRUE)
ENDIF(OSC_INCLUDE_DIR)

FIND_PATH(OSC_INCLUDE_DIR "osc/OscTypes.h"
  PATHS
  $ENV{OSC_HOME}/include
  $ENV{EXTERNLIBS}/oscpack/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "OSC - Headers"
)

SET(OSC_NAMES oscpack)
SET(OSC_DBG_NAMES oscpackd)

FIND_LIBRARY(OSC_LIBRARY NAMES ${OSC_NAMES}
  PATHS
  $ENV{OSC_HOME}
  $ENV{EXTERNLIBS}/oscpack
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "OSC - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OSC_LIBRARY_DEBUG NAMES ${OSC_DBG_NAMES}
    PATHS
    $ENV{OSC_HOME}/lib
    $ENV{EXTERNLIBS}/oscpack/lib
    DOC "OSC - Library (Debug)"
  )
  
  IF(OSC_LIBRARY_DEBUG AND OSC_LIBRARY)
    SET(OSC_LIBRARIES optimized ${OSC_LIBRARY} debug ${OSC_LIBRARY_DEBUG} optimized winmm.lib debug winmm.lib)
  ENDIF(OSC_LIBRARY_DEBUG AND OSC_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSC DEFAULT_MSG OSC_LIBRARY OSC_LIBRARY_DEBUG OSC_INCLUDE_DIR)

  MARK_AS_ADVANCED(OSC_LIBRARY OSC_LIBRARY_DEBUG OSC_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OSC_LIBRARIES ${OSC_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSC DEFAULT_MSG OSC_LIBRARY OSC_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OSC_LIBRARY OSC_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OSC_FOUND)
  SET(OSC_INCLUDE_DIRS ${OSC_INCLUDE_DIR})
ENDIF(OSC_FOUND)
