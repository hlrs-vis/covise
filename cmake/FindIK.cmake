# - Find IK
# Find the IK includes and library
#
#  IK_INCLUDE_DIR - Where to find IK includes
#  IK_LIBRARIES   - List of libraries when using IK
#  IK_FOUND       - True if IK was found

IF(IK_INCLUDE_DIR)
  SET(IK_FIND_QUIETLY TRUE)
ENDIF(IK_INCLUDE_DIR)

FIND_PATH(IK_INCLUDE_DIR "ik/ik.h"
  PATHS
  $ENV{IK_HOME}/include
  $ENV{EXTERNLIBS}/cometIK/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "IK - Headers"
)

SET(IK_NAMES ik)
SET(IK_DBG_NAMES ikd)

FIND_LIBRARY(IK_LIBRARY NAMES ${IK_NAMES}
  PATHS
  $ENV{IK_HOME}
  $ENV{EXTERNLIBS}/cometIK
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "IK - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(IK_LIBRARY_DEBUG NAMES ${IK_DBG_NAMES}
    PATHS
    $ENV{IK_HOME}/lib
    $ENV{EXTERNLIBS}/cometIK/lib
    DOC "IK - Library (Debug)"
  )
  
  IF(IK_LIBRARY_DEBUG AND IK_LIBRARY)
    SET(IK_LIBRARIES optimized ${IK_LIBRARY} debug ${IK_LIBRARY_DEBUG})
  ENDIF(IK_LIBRARY_DEBUG AND IK_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(IK DEFAULT_MSG IK_LIBRARY IK_LIBRARY_DEBUG IK_INCLUDE_DIR)

  MARK_AS_ADVANCED(IK_LIBRARY IK_LIBRARY_DEBUG IK_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(IK_LIBRARIES ${IK_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(IK DEFAULT_MSG IK_LIBRARY IK_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(IK_LIBRARY IK_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(IK_FOUND)
  SET(IK_INCLUDE_DIRS ${IK_INCLUDE_DIR})
ENDIF(IK_FOUND)
