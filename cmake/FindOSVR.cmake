# - Find OSVR
# Find the OSVR includes and library
#
#  OSVR_INCLUDE_DIR - Where to find OSVR includes
#  OSVR_LIBRARIES   - List of libraries when using OSVR
#  OSVR_FOUND       - True if OSVR was found

IF(OSVR_INCLUDE_DIR)
  SET(OSVR_FIND_QUIETLY TRUE)
ENDIF(OSVR_INCLUDE_DIR)

FIND_PATH(OSVR_INCLUDE_DIR "osvr/Client/DisplayConfig.h" "" 
  PATHS
  $ENV{OSVR_HOME}/include
  $ENV{EXTERNLIBS}/OSVR/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "OSVR - Headers"
)

SET(OSVR_NAMES OSVRClient.lib)
SET(OSVR_DBG_NAMES OSVRClientd.lib)
SET(OSVR_RM_NAMES osvrRenderManager.lib)
SET(OSVR_RM_DBG_NAMES osvrRenderManagerd.lib)

FIND_LIBRARY(OSVR_RM_LIBRARY NAMES ${OSVR_RM_NAMES}
  PATHS
  $ENV{OSVR_HOME}
  $ENV{EXTERNLIBS}/OSVR
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 lib/msvc_x64
  DOC "OSVR RenderManager - Library"
)

FIND_LIBRARY(OSVR_LIBRARY NAMES ${OSVR_NAMES}
  PATHS
  $ENV{OSVR_HOME}
  $ENV{EXTERNLIBS}/OSVR
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 lib/msvc_x64
  DOC "OSVR - Library"
)
INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OSVR_LIBRARY_DEBUG NAMES ${OSVR_DBG_NAMES}
    PATHS
    $ENV{OSVR_HOME}/lib
    $ENV{EXTERNLIBS}/OSVR
    PATH_SUFFIXES bin lib lib64 lib/msvc_x64
    DOC "OSVR - Library (Debug)"
  )
  FIND_LIBRARY(OSVR_RM_LIBRARY_DEBUG NAMES ${OSVR_RM_DBG_NAMES}
    PATHS
    $ENV{OSVR_HOME}/lib
    $ENV{EXTERNLIBS}/OSVR
    PATH_SUFFIXES bin lib lib64 lib/msvc_x64
    DOC "OSVR RenderManager - Library (Debug)"
  )
  
  
  IF(OSVR_LIBRARY_DEBUG AND OSVR_LIBRARY)
    SET(OSVR_LIBRARIES optimized ${OSVR_LIBRARY} debug ${OSVR_LIBRARY_DEBUG} optimized ${OSVR_RM_LIBRARY} debug ${OSVR_RM_LIBRARY_DEBUG})
  ENDIF(OSVR_LIBRARY_DEBUG AND OSVR_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSVR DEFAULT_MSG OSVR_LIBRARY OSVR_LIBRARY_DEBUG OSVR_INCLUDE_DIR)

  MARK_AS_ADVANCED(OSVR_LIBRARY OSVR_LIBRARY_DEBUG OSVR_RM_LIBRARY OSVR_RM_LIBRARY_DEBUG   OSVR_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OSVR_LIBRARIES ${OSVR_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSVR DEFAULT_MSG OSVR_LIBRARY OSVR_RM_LIBRARY  OSVR_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OSVR_LIBRARY OSVR_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OSVR_FOUND)
  SET(OSVR_INCLUDE_DIRS ${OSVR_INCLUDE_DIR})
ENDIF(OSVR_FOUND)
