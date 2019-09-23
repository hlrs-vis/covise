# - Find SPEEX
# Find the SPEEX includes and library
#
#  SPEEX_INCLUDE_DIR - Where to find SPEEX includes
#  SPEEX_LIBRARIES   - List of libraries when using SPEEX
#  SPEEX_FOUND       - True if SPEEX was found

IF(SPEEX_INCLUDE_DIR)
  SET(SPEEX_FIND_QUIETLY TRUE)
ENDIF(SPEEX_INCLUDE_DIR)

FIND_PATH(SPEEX_INCLUDE_DIR "speex/speex.h"
  PATHS
  $ENV{SPEEX_HOME}/include
  $ENV{EXTERNLIBS}/speex/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "SPEEX - Headers"
)

SET(SPEEX_NAMES speex libspeex)
SET(SPEEX_DBG_NAMES speexd libspeexd)
SET(SPEEXDSP_NAMES libspeexdsp)
SET(SPEEXDSP_DBG_NAMES libspeexdspd)

FIND_LIBRARY(SPEEX_LIBRARY NAMES ${SPEEX_NAMES}
  PATHS
  $ENV{SPEEX_HOME}
  $ENV{EXTERNLIBS}/speex
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 Lib/x64/VS2012 Lib/Windows/x64/Release/VS2012 Lib/Windows/x64/Debug/VS2012
  DOC "SPEEX - Library"
)

FIND_LIBRARY(SPEEXDSP_LIBRARY NAMES ${SPEEXDSP_NAMES}
  PATHS
  $ENV{SPEEX_HOME}
  $ENV{EXTERNLIBS}/speex
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 Lib/x64/VS2012 Lib/Windows/x64/Release/VS2012 Lib/Windows/x64/Debug/VS2012
  DOC "SPEEX - Library"
)



INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(SPEEX_LIBRARY_DEBUG NAMES ${SPEEX_DBG_NAMES}
    PATHS
  $ENV{SPEEX_HOME}
  $ENV{EXTERNLIBS}/speex
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 Lib/x64/VS2012 Lib/Windows/x64/Debug/VS2012 Lib/Windows/x64/Release/VS2012
    DOC "SPEEX - Library (Debug)"
  )
  FIND_LIBRARY(SPEEXDSP_LIBRARY_DEBUG NAMES ${SPEEXDSP_DBG_NAMES}
    PATHS
  $ENV{SPEEX_HOME}
  $ENV{EXTERNLIBS}/speex
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 Lib/x64/VS2012 Lib/Windows/x64/Debug/VS2012 Lib/Windows/x64/Release/VS2012
    DOC "SPEEX - Library (Debug)"
  )
  
  IF(SPEEX_LIBRARY_DEBUG AND SPEEX_LIBRARY AND SPEEXDSP_LIBRARY_DEBUG AND SPEEXDSP_LIBRARY)
    SET(SPEEX_LIBRARIES optimized ${SPEEX_LIBRARY} debug ${SPEEX_LIBRARY_DEBUG} optimized ${SPEEXDSP_LIBRARY} debug ${SPEEXDSP_LIBRARY_DEBUG})
  ENDIF(SPEEX_LIBRARY_DEBUG AND SPEEX_LIBRARY AND SPEEXDSP_LIBRARY_DEBUG AND SPEEXDSP_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(SPEEX DEFAULT_MSG SPEEX_LIBRARY SPEEX_LIBRARY_DEBUG  SPEEXDSP_LIBRARY SPEEXDSP_LIBRARY_DEBUG  SPEEX_INCLUDE_DIR)

  MARK_AS_ADVANCED(SPEEX_LIBRARY SPEEX_LIBRARY_DEBUG SPEEXDSP_LIBRARY SPEEXDSP_LIBRARY_DEBUG SPEEX_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(SPEEX DEFAULT_MSG SPEEX_LIBRARY SPEEX_INCLUDE_DIR)

  if (SPEEX_FOUND)
      if (SPEEXDSP_LIBRARY)
          set(SPEEX_LIBRARIES ${SPEEX_LIBRARY} ${SPEEXDSP_LIBRARY})
      else()
          set(SPEEX_LIBRARIES ${SPEEX_LIBRARY})
      endif()
  endif()
  
  MARK_AS_ADVANCED(SPEEX_LIBRARY SPEEXDSP_LIBRARY SPEEX_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(SPEEX_FOUND)
  SET(SPEEX_INCLUDE_DIRS ${SPEEX_INCLUDE_DIR})
ENDIF(SPEEX_FOUND)
