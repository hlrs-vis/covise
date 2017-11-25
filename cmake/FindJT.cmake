# - Find JT
# Find the JT includes and library
#
#  JT_INCLUDE_DIR - Where to find JT includes
#  JT_LIBRARIES   - List of libraries when using JT
#  JT_FOUND       - True if JT was found

IF(JT_INCLUDE_DIR)
  SET(JT_FIND_QUIETLY TRUE)
ENDIF(JT_INCLUDE_DIR)

FIND_PATH(JT_INCLUDE_DIR "JtTk/JtkCADImporter.h"
  PATHS
  $ENV{JT_HOME}/include
  $ENV{EXTERNLIBS}/jt/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "JT - Headers"
)

SET(JT_NAMES JtTk40 JtTk43 JtTk85)
SET(JT_DBG_NAMES JtTk40 JtTk43 JtTk85)

FIND_LIBRARY(JT_LIBRARY NAMES ${JT_NAMES}
  PATHS
  $ENV{JT_HOME}
  $ENV{EXTERNLIBS}/jt
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64 lib/win_64_VS2015
  DOC "JT - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(JT_LIBRARY_DEBUG NAMES ${JT_DBG_NAMES}
    PATHS
    $ENV{JT_HOME}/lib
    $ENV{EXTERNLIBS}/jt
    PATH_SUFFIXES lib lib64 lib/win_64_VS2015
    DOC "JT - Library (Debug)"
  )
  
  IF(JT_LIBRARY_DEBUG AND JT_LIBRARY)
    SET(JT_LIBRARIES optimized ${JT_LIBRARY} debug ${JT_LIBRARY_DEBUG})
  ENDIF(JT_LIBRARY_DEBUG AND JT_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(JT DEFAULT_MSG JT_LIBRARY JT_LIBRARY_DEBUG JT_INCLUDE_DIR)

  MARK_AS_ADVANCED(JT_LIBRARY JT_LIBRARY_DEBUG JT_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(JT_LIBRARIES ${JT_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(JT DEFAULT_MSG JT_LIBRARY JT_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(JT_LIBRARY JT_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(JT_FOUND)
  SET(JT_INCLUDE_DIRS ${JT_INCLUDE_DIR})
ENDIF(JT_FOUND)
