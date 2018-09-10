# no default install path exists for NORM, so no standard paths given in here
# NORM needs Protokit libraries, so FindModule for Protokit is called here

IF(NORM_INCLUDE_DIRS AND NORM_LIBRARIES)

  # Already in cache
  set (NORM_FOUND TRUE)

ELSE(NORM_INCLUDE_DIRS AND NORM_LIBRARIES)

  FIND_PACKAGE(Protokit)

  FIND_PATH(NORM_INCLUDE_DIR normApi.h
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
    DOC "The directory where normApi.h resides"
    )

  FIND_LIBRARY(NORM_LIBRARY
    NAMES NORM-1.4b3 norm libnorm.a
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
    DOC "The norm-1.4b3 library" 
    )

  INCLUDE(FindPackageHandleStandardArgs)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(NORM DEFAULT_MSG
      NORM_LIBRARY NORM_INCLUDE_DIR PROTOKIT_LIBRARIES PROTOKIT_INCLUDE_DIR)
	
  IF(NORM_FOUND)
    SET(NORM_LIBRARIES ${NORM_LIBRARY} ${PROTOKIT_LIBRARIES} CACHE STRING "NORM and Protokit libraries" FORCE)
    SET(NORM_INCLUDE_DIRS ${NORM_INCLUDE_DIR} ${PROTOKIT_INCLUDE_DIR} CACHE STRING "NORM and Protokit include directories" FORCE)
  ENDIF(NORM_FOUND)

ENDIF(NORM_INCLUDE_DIRS AND NORM_LIBRARIES)

