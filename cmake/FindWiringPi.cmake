# - Find wiringPi
# Find the wiringPi includes and library
#
#  wiringPi_INCLUDE_DIR - Where to find wiringPi includes
#  wiringPi_LIBRARIES   - List of libraries when using wiringPi
#  wiringPi_FOUND       - True if wiringPi was found

IF(wiringPi_INCLUDE_DIR)
  SET(wiringPi_FIND_QUIETLY TRUE)
ENDIF(wiringPi_INCLUDE_DIR)

FIND_PATH(wiringPi_INCLUDE_DIR "wiringPi.h"
  PATHS
  $ENV{EXTERNLIBS}/wiringPi/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "wiringPi - Headers"
)

SET(wiringPi_NAMES wiringPi wiringPi.lib)
SET(wiringPi_DBG_NAMES wiringPiD libwiringPiD.a wiringPiD.lib)

FIND_LIBRARY(wiringPi_LIBRARY NAMES ${wiringPi_NAMES}
  PATHS
  $ENV{EXTERNLIBS}/wiringPi
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "wiringPi - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(wiringPi_LIBRARY_DEBUG NAMES ${wiringPi_DBG_NAMES}
    PATHS
    $ENV{EXTERNLIBS}/wiringPi/lib
    DOC "wiringPi - Library (Debug)"
  )
  
  IF(wiringPi_LIBRARY_DEBUG AND wiringPi_LIBRARY)
    SET(wiringPi_LIBRARIES optimized ${wiringPi_LIBRARY} debug ${wiringPi_LIBRARY_DEBUG})
  ENDIF(wiringPi_LIBRARY_DEBUG AND wiringPi_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(wiringPi DEFAULT_MSG wiringPi_LIBRARY wiringPi_LIBRARY_DEBUG wiringPi_INCLUDE_DIR)

  MARK_AS_ADVANCED(wiringPi_LIBRARY wiringPi_LIBRARY_DEBUG wiringPi_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(wiringPi_LIBRARIES ${wiringPi_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(wiringPi DEFAULT_MSG wiringPi_LIBRARY wiringPi_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(wiringPi_LIBRARY wiringPi_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(wiringPi_FOUND)
  SET(wiringPi_INCLUDE_DIRS ${wiringPi_INCLUDE_DIR})
ENDIF(wiringPi_FOUND)
