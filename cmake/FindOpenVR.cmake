# - Find OPENVR
# Find the OPENVR includes and library
#
#  OPENVR_INCLUDE_DIR - Where to find OPENVR includes
#  OPENVR_LIBRARIES   - List of libraries when using OPENVR
#  OPENVR_FOUND       - True if OPENVR was found

IF(OPENVR_INCLUDE_DIR)
  SET(OPENVR_FIND_QUIETLY TRUE)
ENDIF(OPENVR_INCLUDE_DIR)

FIND_PATH(OPENVR_INCLUDE_DIR "openvr.h" "openvr_capi.h" 
  PATHS
  $ENV{OPENVR_HOME}/include
  $ENV{EXTERNLIBS}/OpenVR/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "OPENVR - Headers"
  PATH_SUFFIXES openvr
)

SET(OPENVR_NAMES openvr_api openvr_api64.lib openvr_api.lib)
SET(OPENVR_DBG_NAMES openvr_apid openvr_api64d.lib openvr_apid.lib)

FIND_LIBRARY(OPENVR_LIBRARY NAMES ${OPENVR_NAMES}
  PATHS
  $ENV{OPENVR_HOME}
  $ENV{EXTERNLIBS}/OPENVR
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 lib/msvc_x64
  DOC "OPENVR - Library"
)
INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OPENVR_LIBRARY_DEBUG NAMES ${OPENVR_DBG_NAMES}
    PATHS
    $ENV{OPENVR_HOME}/lib
    $ENV{EXTERNLIBS}/OPENVR
    PATH_SUFFIXES bin lib lib64 lib/msvc_x64
    DOC "OPENVR - Library (Debug)"
  )
  
  
  IF(OPENVR_LIBRARY_DEBUG AND OPENVR_LIBRARY)
    SET(OPENVR_LIBRARIES optimized ${OPENVR_LIBRARY} debug ${OPENVR_LIBRARY_DEBUG})
  ENDIF(OPENVR_LIBRARY_DEBUG AND OPENVR_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENVR DEFAULT_MSG OPENVR_LIBRARY OPENVR_LIBRARY_DEBUG OPENVR_INCLUDE_DIR)

  MARK_AS_ADVANCED(OPENVR_LIBRARY OPENVR_LIBRARY_DEBUG   OPENVR_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OPENVR_LIBRARIES ${OPENVR_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENVR DEFAULT_MSG OPENVR_LIBRARY OPENVR_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OPENVR_LIBRARY OPENVR_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OPENVR_FOUND)
  SET(OPENVR_INCLUDE_DIRS ${OPENVR_INCLUDE_DIR})
ENDIF(OPENVR_FOUND)
