# - Find VideoInput
# Find the VideoInput includes and library
#
#  VideoInput_INCLUDE_DIR - Where to find VideoInput includes
#  VideoInput_LIBRARIES   - List of libraries when using VideoInput
#  VideoInput_FOUND       - True if VideoInput was found

IF(VideoInput_INCLUDE_DIR)
  SET(VideoInput_FIND_QUIETLY TRUE)
ENDIF(VideoInput_INCLUDE_DIR)

FIND_PATH(VideoInput_INCLUDE_DIR "videoInput.h"
  PATHS
  $ENV{VideoInput_HOME}/include
  $ENV{EXTERNLIBS}/VideoInput/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "VideoInput - Headers"
)

SET(VideoInput_NAMES videoInput videoInput.lib)
SET(VideoInput_DBG_NAMES videoInputD videoInputD.lib)

FIND_LIBRARY(VideoInput_LIBRARY NAMES ${VideoInput_NAMES}
  PATHS
  $ENV{VideoInput_HOME}
  $ENV{EXTERNLIBS}/VideoInput
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "VideoInput - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(VideoInput_LIBRARY_DEBUG NAMES ${VideoInput_DBG_NAMES}
    PATHS
    $ENV{VideoInput_HOME}/lib
    $ENV{EXTERNLIBS}/VideoInput/lib
    DOC "VideoInput - Library (Debug)"
  )
  
  IF(VideoInput_LIBRARY_DEBUG AND VideoInput_LIBRARY)
    SET(VideoInput_LIBRARIES optimized ${VideoInput_LIBRARY} debug ${VideoInput_LIBRARY_DEBUG} )
  ENDIF(VideoInput_LIBRARY_DEBUG AND VideoInput_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(VideoInput DEFAULT_MSG VideoInput_LIBRARY VideoInput_LIBRARY_DEBUG VideoInput_INCLUDE_DIR)

  MARK_AS_ADVANCED(VideoInput_LIBRARY VideoInput_LIBRARY_DEBUG VideoInput_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(VideoInput_LIBRARIES ${VideoInput_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(VideoInput DEFAULT_MSG VideoInput_LIBRARY VideoInput_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(VideoInput_LIBRARY VideoInput_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(VideoInput_FOUND)
  SET(VideoInput_INCLUDE_DIRS ${VideoInput_INCLUDE_DIR})
ENDIF(VideoInput_FOUND)
