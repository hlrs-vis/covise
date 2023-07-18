# - Find U3D
# Find the U3D includes and library
#
#  U3D_INCLUDE_DIR - Where to find U3D includes
#  U3D_LIBRARIES   - List of libraries when using U3D
#  U3D_FOUND       - True if U3D was found

IF(U3D_INCLUDE_DIR)
  SET(U3D_FIND_QUIETLY TRUE)
ENDIF(U3D_INCLUDE_DIR)

FIND_PATH(U3D_INCLUDE_DIR "U3DHeaders.h"
  PATHS
  $ENV{U3D_HOME}/include
  $ENV{EXTERNLIBS}/U3D/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "U3D - Headers"
)

SET(U3D_NAMES U3D IDTF IDTF.lib)
SET(U3D_DBG_NAMES IDTFD IDTFD.lib)
SET(U3DS_CORE IFXCore IFXCore.lib)
SET(U3DS_DBG_CORE IFXCoreD IFXCoreD.lib)

FIND_LIBRARY(U3D_LIBRARY NAMES ${U3D_NAMES}
  PATHS
  $ENV{U3D_HOME}
  $ENV{EXTERNLIBS}/U3D
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "U3D - Library"
)

FIND_LIBRARY(U3DS_CORE_LIBRARY NAMES ${U3DS_CORE}
  PATHS
  $ENV{U3D_HOME}
  $ENV{EXTERNLIBS}/U3D
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "U3DS - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(U3D_LIBRARY_DEBUG NAMES ${U3D_DBG_NAMES}
    PATHS
    $ENV{U3D_HOME}/lib
    $ENV{EXTERNLIBS}/U3D/lib
    DOC "U3D - Library (Debug)"
  )
  FIND_LIBRARY(U3DS_CORE_LIBRARY_DEBUG NAMES ${U3DS_DBG_NAMES}
    PATHS
    $ENV{U3D_HOME}/lib
    $ENV{EXTERNLIBS}/U3D/lib
    DOC "U3DS - Core Library (Debug)"
  )
  
  IF(U3DS_CORE_LIBRARY AND U3D_LIBRARY)
    SET(U3D_LIBRARIES optimized ${U3D_LIBRARY} debug ${U3D_LIBRARY_DEBUG} optimized ${U3DS_CORE_LIBRARY} debug ${U3DS_CORE_LIBRARY_DEBUG} )
  ENDIF(U3DS_CORE_LIBRARY AND U3D_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(U3D DEFAULT_MSG U3D_LIBRARY U3DS_CORE_LIBRARY U3D_INCLUDE_DIR)

  MARK_AS_ADVANCED(U3D_LIBRARY U3DS_CORE_LIBRARY U3D_LIBRARY_DEBUG U3D_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(U3D_LIBRARIES ${U3D_LIBRARY} ${U3DS_CORE_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(U3D DEFAULT_MSG U3D_LIBRARY U3DS_CORE_LIBRARY U3D_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(U3D_LIBRARY U3DS_CORE_LIBRARY U3D_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(U3D_FOUND)
  SET(U3D_INCLUDE_DIRS ${U3D_INCLUDE_DIR})
ENDIF(U3D_FOUND)
