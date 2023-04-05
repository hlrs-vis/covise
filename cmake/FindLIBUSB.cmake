# - Find LibUSB
# Find the LibUSB includes and library
#
#  LibUSB_INCLUDE_DIR - Where to find LibUSB includes
#  LibUSB_LIBRARIES   - List of libraries when using LibUSB
#  LibUSB_FOUND       - True if LibUSB was found

IF(LIBUSB_INCLUDE_DIR)
  SET(LIBUSB_FIND_QUIETLY TRUE)
ENDIF(LIBUSB_INCLUDE_DIR)

FIND_PATH(LIBUSB_INCLUDE_DIR "lusb0_usb.h" "usb.h" 
  PATHS
  $ENV{LIBUSB_HOME}/include
  $ENV{EXTERNLIBS}/LibUSB/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "LibUSB - Headers"
)

SET(LIBUSB_NAMES usb libusb)
SET(LIBUSB_DBG_NAMES usb libusb)

FIND_LIBRARY(LIBUSB_LIBRARY NAMES ${LIBUSB_NAMES}
  PATHS
  $ENV{LIBUSB_HOME}
  $ENV{EXTERNLIBS}/LibUSB
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES bin lib lib64 lib/msvc_x64
  DOC "LibUSB - Library"
)
INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(LIBUSB_LIBRARY_DEBUG NAMES ${LIBUSB_DBG_NAMES}
    PATHS
    $ENV{LIBUSB_HOME}/lib
    $ENV{EXTERNLIBS}/LibUSB
    PATH_SUFFIXES bin lib lib64 lib/msvc_x64
    DOC "LibUSB - Library (Debug)"
  )
  
  
  IF(LIBUSB_LIBRARY_DEBUG AND LIBUSB_LIBRARY)
    SET(LIBUSB_LIBRARIES optimized ${LIBUSB_LIBRARY} debug ${LIBUSB_LIBRARY_DEBUG})
  ENDIF(LIBUSB_LIBRARY_DEBUG AND LIBUSB_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBUSB DEFAULT_MSG LIBUSB_LIBRARY LIBUSB_LIBRARY_DEBUG LIBUSB_INCLUDE_DIR)

  MARK_AS_ADVANCED(LIBUSB_LIBRARY LIBUSB_LIBRARY_DEBUG   LIBUSB_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(LIBUSB_LIBRARIES ${LIBUSB_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBUSB DEFAULT_MSG LIBUSB_LIBRARY LIBUSB_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(LIBUSB_LIBRARY LIBUSB_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(LIBUSB_FOUND)
  SET(LIBUSB_INCLUDE_DIRS ${LIBUSB_INCLUDE_DIR})
ENDIF(LIBUSB_FOUND)
