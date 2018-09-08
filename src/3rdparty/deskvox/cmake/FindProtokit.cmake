# no default install path exists for protokit, so no standard paths given in here

IF(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)

  # Already in cache
  set (PROTOKIT_FOUND TRUE)

ELSE(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)

  FIND_PATH(PROTOKIT_INCLUDE_DIR protoApp.h
    PATHS 
    $ENV{EXTERNLIBS}/norm/include
    /Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw # Fink
    /opt/local # DarwinPorts
    /opt/csw # Blastwave
    /opt
    DOC "The directory where protokit.h resides"
    )

  FIND_LIBRARY(PROTOKIT_LIBRARY NAMES Protokit libProtokit.a
    PATHS
    $ENV{EXTERNLIBS}/norm
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw
    /opt/local
    /opt/csw
    /opt
    PATH_SUFFIXES lib lib64
    DOC "The Protokit library"
    )

  IF(APPLE)
    SET(EXTRA_LIB "-lresolv")
  ENDIF(APPLE)
  IF(WIN32)
    SET(EXTRA_LIB ws2_32.lib iphlpapi.lib)
  ENDIF(WIN32)
    
  SET(PROTOKIT_LIBRARIES ${PROTOKIT_LIBRARY} ${EXTRA_LIB} CACHE STRING "The Protokit libraries")

  INCLUDE(FindPackageHandleStandardArgs)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(Protokit DEFAULT_MSG PROTOKIT_LIBRARIES PROTOKIT_INCLUDE_DIR)

ENDIF(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)
