# - Find E57
# Find the E57 includes and library
#
#  E57_INCLUDE_DIR - Where to find E57 includes
#  E57_LIBRARIES   - List of libraries when using E57
#  E57_FOUND       - True if E57 was found

IF(E57_INCLUDE_DIR)
  SET(E57_FIND_QUIETLY TRUE)
ENDIF(E57_INCLUDE_DIR)

FIND_PATH(E57_INCLUDE_DIR "e57/E57Foundation.h"
  PATHS
  $ENV{EXTERNLIBS}/e57/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "E57 - Headers"
)

SET(E57_NAMES e57 libE57RefImpl.a E57RefImpl.lib)
SET(E57_DBG_NAMES e57D libE57RefImplD.a E57RefImpl-d.lib)

FIND_LIBRARY(E57_LIBRARY NAMES ${E57_NAMES}
  PATHS
  $ENV{EXTERNLIBS}/e57
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "E57 - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(E57_LIBRARY_DEBUG NAMES ${E57_DBG_NAMES}
    PATHS
    $ENV{EXTERNLIBS}/e57/lib
    DOC "E57 - Library (Debug)"
  )
  
  IF(E57_LIBRARY_DEBUG AND E57_LIBRARY)
    SET(E57_LIBRARIES optimized ${E57_LIBRARY} debug ${E57_LIBRARY_DEBUG})
  ENDIF(E57_LIBRARY_DEBUG AND E57_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(E57 DEFAULT_MSG E57_LIBRARY E57_LIBRARY_DEBUG E57_INCLUDE_DIR)

  MARK_AS_ADVANCED(E57_LIBRARY E57_LIBRARY_DEBUG E57_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(E57_LIBRARIES ${E57_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(E57 DEFAULT_MSG E57_LIBRARY E57_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(E57_LIBRARY E57_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(E57_FOUND)
  SET(E57_INCLUDE_DIRS ${E57_INCLUDE_DIR})
ENDIF(E57_FOUND)
