# - Find SoQt Library
# This module defines the following variables
#  SOQT_FOUND           - system has soqt
#  SOQT_INCLUDE_DIRS    - where the soqt include directory can be found
#  SOQT_LIBRARIES       - link to this to use soqt

IF (WIN32)
  IF (CYGWIN OR MINGW)

    FIND_PATH(SOQT_INCLUDE_DIRS Inventor/Qt/SoQt.h
      PATHS $ENV{SOQT_HOME}/include $ENV{EXTERNLIBS}/SoQt/include $ENV{EXTERNLIBS}/Coin3D/include $ENV{COINDIR}/include
      NO_DEFAULT_PATH
    )
    FIND_PATH(SOQT_INCLUDE_DIRS Inventor/Qt/SoQt.h)

    FIND_LIBRARY(SOQT_LIBRARIES SoQt2 SoQt1 SoQt
      PATHS $ENV{SOQT_HOME}/lib $ENV{EXTERNLIBS}/SoQt/lib $ENV{EXTERNLIBS}/Coin3D/lib $ENV{COINDIR}/lib
      NO_DEFAULT_PATH
    )
    FIND_LIBRARY(SOQT_LIBRARIES SoQt2 SoQt1 SoQt)

  ELSE (CYGWIN OR MINGW)

    FIND_PATH(SOQT_INCLUDE_DIRS Inventor/Qt/SoQt.h
      PATHS $ENV{SOQT_HOME}/include $ENV{EXTERNLIBS}/SoQt/include $ENV{EXTERNLIBS}/Coin3D/include $ENV{COINDIR}/include
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\Coin3D\\2;Installation Path]/include"
    )

    FIND_LIBRARY(SOQT_LIBRARY_DEBUG SoQt2d SoQt1d SoQtd
      PATHS $ENV{SOQT_HOME}/lib $ENV{EXTERNLIBS}/SoQt/lib $ENV{EXTERNLIBS}/Coin3D/lib $ENV{COINDIR}/lib
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\Coin3D\\2;Installation Path]/lib"
    )

    FIND_LIBRARY(SOQT_LIBRARY_RELEASE SoQt2 SoQt1 SoQt
      PATHS $ENV{SOQT_HOME}/lib $ENV{EXTERNLIBS}/SoQt/lib $ENV{EXTERNLIBS}/Coin3D/lib $ENV{COINDIR}/lib
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\Coin3D\\2;Installation Path]/lib"
    )

    IF (SOQT_LIBRARY_DEBUG AND SOQT_LIBRARY_RELEASE)
      SET(SOQT_LIBRARIES optimized ${SOQT_LIBRARY_RELEASE} debug ${SOQT_LIBRARY_DEBUG})
    ELSE (SOQT_LIBRARY_DEBUG AND SOQT_LIBRARY_RELEASE)
      IF (SOQT_LIBRARY_DEBUG)
        SET (SOQT_LIBRARIES ${SOQT_LIBRARY_DEBUG})
      ENDIF (SOQT_LIBRARY_DEBUG)
      IF (SOQT_LIBRARY_RELEASE)
        SET (SOQT_LIBRARIES ${SOQT_LIBRARY_RELEASE})
      ENDIF (SOQT_LIBRARY_RELEASE)
    ENDIF (SOQT_LIBRARY_DEBUG AND SOQT_LIBRARY_RELEASE)

  ENDIF (CYGWIN OR MINGW)

ELSE (WIN32)
  IF(APPLE)
    FIND_PATH(SOQT_INCLUDE_DIRS SoQt.h
      PATHS
      $ENV{EXTERNLIBS}/SoQt.framework/Headers
      /Library/Frameworks/SoQt.framework/Headers 
    )
    FIND_LIBRARY(SOQT_LIBRARIES SoQt
      PATHS
      $ENV{EXTERNLIBS}/SoQt.framework/Libraries
      /Library/Frameworks/SoQt.framework/Libraries
    )   
    SET(SOQT_LIBRARIES "-framework SoQt" CACHE STRING "SoQt library for OSX")
  ELSE(APPLE)

    FIND_PATH(SOQT_INCLUDE_DIRS Inventor/Qt/SoQt.h
      PATHS $ENV{SOQT_HOME}/include $ENV{SOQT_HOME}/include/Coin2
      $ENV{EXTERNLIBS}/SoQt/include $ENV{EXTERNLIBS}/Coin3D/include
      $ENV{COINDIR}/include /usr/include/Coin2 /usr/include/coin
      NO_DEFAULT_PATH
    )
    FIND_PATH(SOQT_INCLUDE_DIRS Inventor/Qt/SoQt.h)

    FIND_LIBRARY(SOQT_LIBRARIES SoQt4 SoQt2 SoQt1 SoQt
      PATHS $ENV{SOQT_HOME}/lib $ENV{EXTERNLIBS}/SoQt/lib $ENV{EXTERNLIBS}/Coin3D/lib $ENV{COINDIR}/lib
    )
    FIND_LIBRARY(SOQT_LIBRARIES SoQt4 SoQt2 SoQt1 SoQt)


  ENDIF(APPLE)

ENDIF (WIN32)

# handle the QUIETLY and REQUIRED arguments and set SOQT_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SoQt DEFAULT_MSG SOQT_LIBRARIES SOQT_INCLUDE_DIRS)

MARK_AS_ADVANCED(SOQT_INCLUDE_DIRS SOQT_LIBRARIES SOQT_LIBRARY_RELEASE SOQT_LIBRARY_DEBUG)
