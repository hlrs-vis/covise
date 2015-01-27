# - Find OVR
# Find the OVR includes and library
#
#  OVR_INCLUDE_DIR - Where to find OVR includes
#  OVR_LIBRARIES   - List of libraries when using OVR
#  OVR_FOUND       - True if OVR was found

IF(OVR_INCLUDE_DIR)
  SET(OVR_FIND_QUIETLY TRUE)
ENDIF(OVR_INCLUDE_DIR)

FIND_PATH(OVR_INCLUDE_DIR "OVR.h"
  PATHS
  $ENV{OVR_HOME}/include
  $ENV{EXTERNLIBS}/OculusSDK/LibOVR/include
  $ENV{EXTERNLIBS}/LibOVR/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "OVR - Headers"
)

SET(OVR_NAMES libovr64)
SET(OVR_DBG_NAMES libovr64d)

FIND_LIBRARY(OVR_LIBRARY NAMES ${OVR_NAMES}
  PATHS
  $ENV{OVR_HOME}
  $ENV{EXTERNLIBS}/OculusSDK/LibOVR
  $ENV{EXTERNLIBS}/LibOVR
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 Lib/x64/VS2012
  DOC "OVR - Library"
)


INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OVR_LIBRARY_DEBUG NAMES ${OVR_DBG_NAMES}
    PATHS
  $ENV{OVR_HOME}
  $ENV{EXTERNLIBS}/OculusSDK/LibOVR
  $ENV{EXTERNLIBS}/LibOVR
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 Lib/x64/VS2012
    DOC "OVR - Library (Debug)"
  )
  
  IF(OVR_LIBRARY_DEBUG AND OVR_LIBRARY)
    SET(OVR_LIBRARIES optimized ${OVR_LIBRARY} debug ${OVR_LIBRARY_DEBUG})
  ENDIF(OVR_LIBRARY_DEBUG AND OVR_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OVR DEFAULT_MSG OVR_LIBRARY OVR_LIBRARY_DEBUG  OVR_INCLUDE_DIR)

  MARK_AS_ADVANCED(OVR_LIBRARY OVR_LIBRARY_DEBUG OVR_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OVR_LIBRARIES ${OVR_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OVR DEFAULT_MSG OVR_LIBRARY OVR_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OVR_LIBRARY OVR_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OVR_FOUND)
  SET(OVR_INCLUDE_DIRS ${OVR_INCLUDE_DIR})
ENDIF(OVR_FOUND)
