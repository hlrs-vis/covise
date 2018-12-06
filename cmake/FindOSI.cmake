# - Find OSI
# Find the OSI includes and library
#
#  OSI_INCLUDE_DIR - Where to find OSI includes
#  OSI_LIBRARIES   - List of libraries when using OSI
#  OSI_FOUND       - True if OSI was found

IF(OSI_INCLUDE_DIR)
  SET(OSI_FIND_QUIETLY TRUE)
ENDIF(OSI_INCLUDE_DIR)

FIND_PATH(OSI_INCLUDE_DIR "osi/osi_version.pb.h"
  PATHS
  $ENV{OSI_HOME}/include
  $ENV{EXTERNLIBS}/OSI/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "OSI - Headers"
)

SET(OSI_NAMES osi3/open_simulation_interface_pic.lib osi/open_simulation_interface_pic.lib)
SET(OSI_DBG_NAMES osi3/open_simulation_interface_picd.lib osi/open_simulation_interface_picd.lib)

FIND_LIBRARY(OSI_LIBRARY NAMES ${OSI_NAMES}
  PATHS
  $ENV{OSI_HOME}
  $ENV{EXTERNLIBS}/OSI
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "OSI - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OSI_LIBRARY_DEBUG NAMES ${OSI_DBG_NAMES}
    PATHS
    $ENV{OSI_HOME}/lib
    $ENV{EXTERNLIBS}/OSI/lib
    DOC "OSI - Library (Debug)"
  )
  
  IF(OSI_LIBRARY_DEBUG AND OSI_LIBRARY)
    SET(OSI_LIBRARIES optimized ${OSI_LIBRARY} debug ${OSI_LIBRARY_DEBUG})
  ENDIF(OSI_LIBRARY_DEBUG AND OSI_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSI DEFAULT_MSG OSI_LIBRARY OSI_LIBRARY_DEBUG OSI_INCLUDE_DIR)

  MARK_AS_ADVANCED(OSI_LIBRARY OSI_LIBRARY_DEBUG OSI_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OSI_LIBRARIES ${OSI_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OSI DEFAULT_MSG OSI_LIBRARY OSI_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OSI_LIBRARY OSI_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OSI_FOUND)
  SET(OSI_INCLUDE_DIRS ${OSI_INCLUDE_DIR})
ENDIF(OSI_FOUND)
