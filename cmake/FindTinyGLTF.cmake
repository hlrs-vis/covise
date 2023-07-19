# - Find TinyGLTF
# Find the TinyGLTF includes and library
#
#  TinyGLTF_INCLUDE_DIR - Where to find TinyGLTF includes
#  TinyGLTF_FOUND       - True if TinyGLTF was found

IF(TinyGLTF_INCLUDE_DIR)
  SET(TinyGLTF_FIND_QUIETLY TRUE)
ENDIF(TinyGLTF_INCLUDE_DIR)

FIND_PATH(TinyGLTF_INCLUDE_DIR "tiny_gltf.h"
  PATHS
  $ENV{TinyGLTF_HOME}/include
  $ENV{EXTERNLIBS}/TinyGLTF/include
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  DOC "TinyGLTF - Headers"
)

SET(TinyGLTF_NAMES TinyGLTF tinygltf)
SET(TinyGLTF_DBG_NAMES TinyGLTFd)

FIND_LIBRARY(TinyGLTF_LIBRARY NAMES ${TinyGLTF_NAMES}
  PATHS
  $ENV{TinyGLTF_HOME}
  $ENV{EXTERNLIBS}/TinyGLTF
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "TinyGLTF - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(TinyGLTF_LIBRARY_DEBUG NAMES ${TinyGLTF_DBG_NAMES}
    PATHS
    $ENV{TinyGLTF_HOME}
    $ENV{EXTERNLIBS}/TinyGLTF
    PATH_SUFFIXES lib lib64
    DOC "TinyGLTF - Library (Debug)"
  )
  
  IF(TinyGLTF_LIBRARY_DEBUG AND TinyGLTF_LIBRARY)
    SET(TinyGLTF_LIBRARIES optimized ${TinyGLTF_LIBRARY} debug ${TinyGLTF_LIBRARY_DEBUG})
  ENDIF(TinyGLTF_LIBRARY_DEBUG AND TinyGLTF_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(TinyGLTF DEFAULT_MSG TinyGLTF_LIBRARY TinyGLTF_LIBRARY_DEBUG TinyGLTF_INCLUDE_DIR)

  MARK_AS_ADVANCED(TinyGLTF_LIBRARY TinyGLTF_LIBRARY_DEBUG TinyGLTF_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(TinyGLTF_LIBRARIES ${TinyGLTF_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(TinyGLTF DEFAULT_MSG TinyGLTF_LIBRARY TinyGLTF_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(TinyGLTF_LIBRARY TinyGLTF_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(TinyGLTF_FOUND)
  SET(TinyGLTF_INCLUDE_DIRS ${TinyGLTF_INCLUDE_DIR})
ENDIF(TinyGLTF_FOUND)
