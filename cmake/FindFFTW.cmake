# - Find FFTW
# Find the FFTW includes and library
#
#  FFTW_INCLUDE_DIR - Where to find FFTW includes
#  FFTW_LIBRARIES   - List of libraries when using FFTW
#  FFTW_FOUND       - True if FFTW was found

IF(FFTW_INCLUDE_DIR)
  SET(FFTW_FIND_QUIETLY TRUE)
ENDIF(FFTW_INCLUDE_DIR)

FIND_PATH(FFTW_INCLUDE_DIR "fftw3.h"
  PATHS
  $ENV{FFTW_HOME}/include
  $ENV{EXTERNLIBS}/fftw/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "FFTW - Headers"
)

SET(FFTW_NAMES fftw3 fftw-3.3 libfftwf-3.3 fftw3-3 libfftw3-3.lib)
SET(FFTW_DBG_NAMES fftw3 fftw3-3 libfftwf-3.3d fftw-3.3d libfftw3-3.lib)

FIND_LIBRARY(FFTW_LIBRARY NAMES ${FFTW_NAMES}
  PATHS
  $ENV{FFTW_HOME}
  $ENV{EXTERNLIBS}/fftw
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "FFTW - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(FFTW_LIBRARY_DEBUG NAMES ${FFTW_DBG_NAMES}
    PATHS
    $ENV{FFTW_HOME}/lib
    $ENV{EXTERNLIBS}/fftw/lib
    DOC "FFTW - Library (Debug)"
  )
  
  IF(FFTW_LIBRARY_DEBUG AND FFTW_LIBRARY)
    SET(FFTW_LIBRARIES optimized ${FFTW_LIBRARY} debug ${FFTW_LIBRARY_DEBUG})
  ENDIF(FFTW_LIBRARY_DEBUG AND FFTW_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFTW DEFAULT_MSG FFTW_LIBRARY FFTW_LIBRARY_DEBUG FFTW_INCLUDE_DIR)

  MARK_AS_ADVANCED(FFTW_LIBRARY FFTW_LIBRARY_DEBUG FFTW_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(FFTW_LIBRARIES ${FFTW_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFTW DEFAULT_MSG FFTW_LIBRARY FFTW_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(FFTW_LIBRARY FFTW_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(FFTW_FOUND)
  SET(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})
ENDIF(FFTW_FOUND)
