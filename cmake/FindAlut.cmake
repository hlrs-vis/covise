# - Find Alut
# Find the alut includes and library
#
#  ALUT_INCLUDE_DIR - Where to find alut includes
#  ALUT_LIBRARIES   - List of libraries when using alut
#  ALUT_FOUND       - True if alut was found

IF(ALUT_INCLUDE_DIR)
  SET(ALUT_FIND_QUIETLY TRUE)
ENDIF(ALUT_INCLUDE_DIR)

IF(APPLE OR MSVC)
  SET(ALUT_H_NAME "alut.h")
ELSE(APPLE OR MSVC)
  SET(ALUT_H_NAME "AL/alut.h")
ENDIF(APPLE OR MSVC)

FIND_PATH(ALUT_INCLUDE_DIR ${ALUT_H_NAME}
  PATHS
  $ENV{ALUT_HOME}/include
  $ENV{EXTERNLIBS}/alut/include
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
  PATH_SUFFIXES AL include/AL include/OpenAL include
  DOC "alut - Headers"
)

SET(ALUT_NAMES alut)
SET(ALUT_DBG_NAMES alutd)

FIND_LIBRARY(ALUT_LIBRARY NAMES ${ALUT_NAMES}
  PATHS
  $ENV{ALUT_HOME}/lib
  $ENV{EXTERNLIBS}/alut/lib
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
  DOC "alut - Library"
)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(ALUT_LIBRARY_DEBUG NAMES ${ALUT_DBG_NAMES}
    PATHS
    $ENV{ALUT_HOME}/lib
    $ENV{EXTERNLIBS}/alut/lib
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
    DOC "alut - Library (Debug)"
  )
  
  IF(ALUT_LIBRARY_DEBUG AND ALUT_LIBRARY)
    SET(ALUT_LIBRARIES optimized ${ALUT_LIBRARY} debug ${ALUT_LIBRARY_DEBUG})
  ENDIF(ALUT_LIBRARY_DEBUG AND ALUT_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(ALUT DEFAULT_MSG ALUT_LIBRARY ALUT_LIBRARY_DEBUG ALUT_INCLUDE_DIR)

  MARK_AS_ADVANCED(ALUT_LIBRARY ALUT_LIBRARY_DEBUG ALUT_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(ALUT_LIBRARIES ${ALUT_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(ALUT DEFAULT_MSG ALUT_LIBRARY ALUT_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(ALUT_LIBRARY ALUT_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(ALUT_FOUND)
  SET(ALUT_INCLUDE_DIRS ${ALUT_INCLUDE_DIR})
ENDIF(ALUT_FOUND)
