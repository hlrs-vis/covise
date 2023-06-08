# - Find PCL
# Find the PCL includes and library
#
#  PCL_INCLUDE_DIR - Where to find PCL includes
#  PCL_LIBRARIES   - List of libraries when using PCL
#  PCL_FOUND       - True if PCL was found

IF(PCL_INCLUDE_DIR)
  SET(PCL_FIND_QUIETLY TRUE)
ENDIF(PCL_INCLUDE_DIR)

FIND_PATH(PCL_INCLUDE_DIR "pcl/pcl_base.h"
  PATHS
  $ENV{PCL_HOME}/include
  $ENV{EXTERNLIBS}/pcl/include
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local/include
  /usr/include
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  PATH_SUFFIXES pcl-1.15 pcl-1.14 pcl-1.13 pcl-1.12 pcl-1.11 pcl-1.10 pcl-1.9 pcl-1.8 pcl-1.7 pcl
  DOC "PCL - Headers"
)

SET(PCL_NAMES pcl_common_release pcl_common libpcl_common)
SET(PCL_DBG_NAMES pcl_common_debug)
SET(PCL_IO_NAMES pcl_io_release pcl_io libpcl_io)
SET(PCL_IO_DBG_NAMES pcl_io_debug)
SET(PCL_OOC_NAMES pcl_outofcore_release pcl_outofcore libpcl_outofcore)
SET(PCL_OOC_DBG_NAMES pcl_outofcore_debug)

FIND_LIBRARY(PCL_LIBRARY NAMES ${PCL_NAMES}
  PATHS
  $ENV{PCL_HOME}
  $ENV{EXTERNLIBS}/pcl
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "PCL - Library"
)

FIND_LIBRARY(PCL_IO_LIBRARY NAMES ${PCL_IO_NAMES}
  PATHS
  $ENV{PCL_HOME}
  $ENV{EXTERNLIBS}/pcl
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "PCL-IO - Library"
)

FIND_LIBRARY(PCL_OOC_LIBRARY NAMES ${PCL_OOC_NAMES}
  PATHS
  $ENV{PCL_HOME}
  $ENV{EXTERNLIBS}/pcl
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "PCL-OOC - Library"
)
INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(PCL_LIBRARY_DEBUG NAMES ${PCL_DBG_NAMES}
    PATHS
    $ENV{PCL_HOME}
    $ENV{EXTERNLIBS}/pcl
    PATH_SUFFIXES lib lib64
    DOC "PCL - Library (Debug)"
  )
  FIND_LIBRARY(PCL_IO_LIBRARY_DEBUG NAMES ${PCL_IO_DBG_NAMES}
    PATHS
    $ENV{PCL_HOME}
    $ENV{EXTERNLIBS}/pcl
    PATH_SUFFIXES lib lib64
    DOC "PCL - IO - Library (Debug)"
  )
  FIND_LIBRARY(PCL_OOC_LIBRARY_DEBUG NAMES ${PCL_OOC_DBG_NAMES}
    PATHS
    $ENV{PCL_HOME}
    $ENV{EXTERNLIBS}/pcl
    PATH_SUFFIXES lib lib64
    DOC "PCL - OOC - Library (Debug)"
  )
  
  IF(PCL_LIBRARY_DEBUG AND PCL_LIBRARY)
    SET(PCL_LIBRARIES optimized ${PCL_LIBRARY} debug ${PCL_LIBRARY_DEBUG} optimized ${PCL_IO_LIBRARY} debug ${PCL_IO_LIBRARY_DEBUG})
  ENDIF(PCL_LIBRARY_DEBUG AND PCL_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(PCL DEFAULT_MSG PCL_LIBRARY PCL_LIBRARY_DEBUG PCL_IO_LIBRARY PCL_IO_LIBRARY_DEBUG PCL_OOC_LIBRARY PCL_OOC_LIBRARY_DEBUG PCL_INCLUDE_DIR)

  MARK_AS_ADVANCED(PCL_LIBRARY PCL_LIBRARY_DEBUG PCL_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(PCL_LIBRARIES ${PCL_LIBRARY} ${PCL_IO_LIBRARY} ${PCL_OOC_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(PCL DEFAULT_MSG PCL_LIBRARY PCL_IO_LIBRARY PCL_OOC_LIBRARY PCL_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(PCL_LIBRARY PCL_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(PCL_FOUND)
  SET(PCL_INCLUDE_DIRS ${PCL_INCLUDE_DIR})
ENDIF(PCL_FOUND)
