# - Find LibUSB
# Find the LibUSB includes and library
#
#  LibUSB_INCLUDE_DIR - Where to find LibUSB includes
#  LibUSB_LIBRARIES   - List of libraries when using LibUSB
#  LibUSB_FOUND       - True if LibUSB was found

IF(LIBUSB1_INCLUDE_DIR)
  SET(LIBUSB1_FIND_QUIETLY TRUE)
ENDIF(LIBUSB1_INCLUDE_DIR)

FIND_PATH(LIBUSB1_INCLUDE_DIR "libusb-1.0/libusb.h" 
  PATHS
  $ENV{LIBUSB1_HOME}/include
  $ENV{EXTERNLIBS}/libusb1/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "LibUSB1 - Headers"
)

SET(LIBUSB1_NAMES libusb-1.0 usb-1.0)
SET(LIBUSB1_DBG_NAMES libusb-1.0d usb-1.0d)

FIND_LIBRARY(LIBUSB1_LIBRARY NAMES ${LIBUSB1_NAMES}
  PATHS
  $ENV{LIBUSB1_HOME}
  $ENV{EXTERNLIBS}/libusb1
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 lib/msvc_x64
  DOC "LibUSB1 - Library"
)
INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(LIBUSB1_LIBRARY_DEBUG NAMES ${LIBUSB1_DBG_NAMES}
    PATHS
    $ENV{LIBUSB1_HOME}/lib
    $ENV{EXTERNLIBS}/libusb1
    PATH_SUFFIXES bin lib lib64 lib/msvc_x64
    DOC "LibUSB1 - Library (Debug)"
  )
  
  
  IF(LIBUSB1_LIBRARY_DEBUG AND LIBUSB1_LIBRARY)
    SET(LIBUSB1_LIBRARIES optimized ${LIBUSB1_LIBRARY} debug ${LIBUSB1_LIBRARY_DEBUG})
  ENDIF(LIBUSB1_LIBRARY_DEBUG AND LIBUSB1_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBUSB1 DEFAULT_MSG LIBUSB1_LIBRARY LIBUSB1_LIBRARY_DEBUG LIBUSB1_INCLUDE_DIR)

  MARK_AS_ADVANCED(LIBUSB1_LIBRARY LIBUSB1_LIBRARY_DEBUG   LIBUSB1_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(LIBUSB1_LIBRARIES ${LIBUSB1_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBUSB1 DEFAULT_MSG LIBUSB1_LIBRARY LIBUSB1_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(LIBUSB1_LIBRARY LIBUSB1_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(LIBUSB1_FOUND)
  SET(LIBUSB1_INCLUDE_DIRS ${LIBUSB1_INCLUDE_DIR})
ENDIF(LIBUSB1_FOUND)
