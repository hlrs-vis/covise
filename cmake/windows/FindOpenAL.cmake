# - Find OpenAL
# Find the OpenAL includes and library
#
#  OPENAL_INCLUDE_DIR - Where to find OpenAL includes
#  OPENAL_LIBRARIES   - List of libraries when using OpenAL
#  OPENAL_FOUND       - True if OpenAL was found

IF(OPENAL_INCLUDE_DIR)
  SET(OPENAL_FIND_QUIETLY TRUE)
ENDIF(OPENAL_INCLUDE_DIR)

FIND_PATH(OPENAL_INCLUDE_DIR al.h
  PATHS
  $ENV{OPENAL_HOME}/include
  $ENV{EXTERNLIBS}/OpenAL/include
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  [HKEY_LOCAL_MACHINE\\SOFTWARE\\Creative\ Labs\\OpenAL\ 1.1\ Software\ Development\ Kit\\1.00.0000;InstallDir]
  PATH_SUFFIXES include/AL include/OpenAL include
  DOC "OpenAL - Headers"
)

SET(OPENAL_NAMES OpenAL al openal OpenAL32)
SET(OPENAL_DBG_NAMES OpenALd ald openald OpenAL32d)

FIND_LIBRARY(OPENAL_LIBRARY NAMES ${OPENAL_NAMES}
  PATHS
  $ENV{OPENAL_HOME}/lib
  $ENV{EXTERNLIBS}/OpenAL/lib
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  [HKEY_LOCAL_MACHINE\\SOFTWARE\\Creative\ Labs\\OpenAL\ 1.1\ Software\ Development\ Kit\\1.00.0000;InstallDir]
  PATH_SUFFIXES lib64 lib libs64 libs libs/Win32 libs/Win64
  DOC "OpenAL - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(OPENAL_LIBRARY_DEBUG NAMES ${OPENAL_DBG_NAMES}
    PATHS
    $ENV{OPENAL_HOME}/lib
    $ENV{EXTERNLIBS}/OpenAL/lib
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw
    /opt/local
    /opt/csw
    /opt
    [HKEY_LOCAL_MACHINE\\SOFTWARE\\Creative\ Labs\\OpenAL\ 1.1\ Software\ Development\ Kit\\1.00.0000;InstallDir]
    PATH_SUFFIXES lib64 lib libs64 libs libs/Win32 libs/Win64
    DOC "OpenAL - Library (Debug)"
  )
  
  IF(OPENAL_LIBRARY_DEBUG AND OPENAL_LIBRARY)
    SET(OPENAL_LIBRARIES optimized ${OPENAL_LIBRARY} debug ${OPENAL_LIBRARY_DEBUG})
  ENDIF(OPENAL_LIBRARY_DEBUG AND OPENAL_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENAL DEFAULT_MSG OPENAL_LIBRARY OPENAL_LIBRARY_DEBUG OPENAL_INCLUDE_DIR)

  MARK_AS_ADVANCED(OPENAL_LIBRARY OPENAL_LIBRARY_DEBUG OPENAL_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(OPENAL_LIBRARIES ${OPENAL_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENAL DEFAULT_MSG OPENAL_LIBRARY OPENAL_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OPENAL_LIBRARY OPENAL_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(OPENAL_FOUND)
  SET(OPENAL_INCLUDE_DIRS ${OPENAL_INCLUDE_DIR})
ENDIF(OPENAL_FOUND)
