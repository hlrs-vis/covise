# - Find WIRINGPI
# Find the wiringPi includes and library
#
#  WIRINGPI_INCLUDE_DIR - Where to find wiringPi includes
#  WIRINGPI_LIBRARIES   - List of libraries when using wiringPi
#  WIRINGPI_FOUND       - True if wiringPi was found

IF(WIRINGPI_INCLUDE_DIR)
  SET(WIRINGPI_FIND_QUIETLY TRUE)
ENDIF(WIRINGPI_INCLUDE_DIR)

FIND_PATH(WIRINGPI_INCLUDE_DIR "wiringPi.h"
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

SET(WIRINGPI_NAMES wiringPi wiringPi.lib)
SET(WIRINGPI_DBG_NAMES wiringPiD libwiringPiD.a wiringPiD.lib)

FIND_LIBRARY(WIRINGPI_LIBRARY NAMES ${WIRINGPI_NAMES}
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
  FIND_LIBRARY(WIRINGPI_LIBRARY_DEBUG NAMES ${WIRINGPI_DBG_NAMES}
    PATHS
    $ENV{EXTERNLIBS}/wiringPi/lib
    DOC "wiringPi - Library (Debug)"
  )
  
  IF(WIRINGPI_LIBRARY_DEBUG AND WIRINGPI_LIBRARY)
    SET(WIRINGPI_LIBRARIES optimized ${WIRINGPI_LIBRARY} debug ${WIRINGPI_LIBRARY_DEBUG})
  ENDIF(WIRINGPI_LIBRARY_DEBUG AND WIRINGPI_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(WIRINGPI DEFAULT_MSG WIRINGPI_LIBRARY WIRINGPI_LIBRARY_DEBUG WIRINGPI_INCLUDE_DIR)

  MARK_AS_ADVANCED(WIRINGPI_LIBRARY WIRINGPI_LIBRARY_DEBUG WIRINGPI_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(WIRINGPI_LIBRARIES ${WIRINGPI_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(WIRINGPI DEFAULT_MSG WIRINGPI_LIBRARY WIRINGPI_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(WIRINGPI_LIBRARY WIRINGPI_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(WIRINGPI_FOUND)
  SET(WIRINGPI_INCLUDE_DIRS ${WIRINGPI_INCLUDE_DIR})
ENDIF(WIRINGPI_FOUND)
