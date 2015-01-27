# - try to find glut library and include files
#  GLUT_INCLUDE_DIR, where to find GL/glut.h, etc.
#  GLUT_LIBRARIES, the libraries to link against
#  GLUT_FOUND, If false, do not try to use GLUT.
# Also defined, but not for general use are:
#  GLUT_glut_LIBRARY = the full path to the glut library.
#  GLUT_Xmu_LIBRARY  = the full path to the Xmu library.
#  GLUT_Xi_LIBRARY   = the full path to the Xi Library.

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

IF (WIN32)

  SET(GLUT_PACKAGE_DEFINITIONS "-DGLUT_NO_LIB_PRAGMA")
  FIND_PATH( GLUT_INCLUDE_DIR NAMES GL/glut.h 
    PATHS  
    $ENV{EXTERNLIBS}/glut/include
    ${GLUT_ROOT_PATH}/include )
  FIND_LIBRARY( GLUT_LIBRARY_DEBUG NAMES glutD glut32D freeglutD
    PATHS
    $ENV{EXTERNLIBS}/glut/lib
    ${OPENGL_LIBRARY_DIR}
    ${GLUT_ROOT_PATH}/Debug
    )
    FIND_LIBRARY( GLUT_LIBRARY_RELEASE NAMES glut glut32 freeglut
    PATHS
    ${OPENGL_LIBRARY_DIR}
    $ENV{EXTERNLIBS}/glut/lib
    ${GLUT_ROOT_PATH}/Release
    )
    IF(MSVC_IDE)
      IF (GLUT_LIBRARY_DEBUG AND GLUT_LIBRARY_RELEASE)
         SET(GLUT_LIBRARY optimized ${GLUT_LIBRARY_RELEASE} debug ${GLUT_LIBRARY_DEBUG})
      ELSE (GLUT_LIBRARY_DEBUG AND GLUT_LIBRARY_RELEASE)
         SET(GLUT_LIBRARY NOTFOUND)
         MESSAGE(STATUS "Could not find the debug AND release version of zlib")
      ENDIF (GLUT_LIBRARY_DEBUG AND GLUT_LIBRARY_RELEASE)
    ELSE(MSVC_IDE)
      STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_TOLOWER)
      IF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(GLUT_LIBRARY ${GLUT_LIBRARY_DEBUG})
      ELSE(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(GLUT_LIBRARY ${GLUT_LIBRARY_RELEASE})
      ENDIF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
    ENDIF(MSVC_IDE)
    MARK_AS_ADVANCED(GLUT_LIBRARY_DEBUG GLUT_LIBRARY_RELEASE)
ELSE (WIN32)
  
  IF (APPLE)
    # These values for Apple could probably do with improvement.
    FIND_PATH( GLUT_INCLUDE_DIR glut.h
      /System/Library/Frameworks/GLUT.framework/Versions/A/Headers
      ${OPENGL_LIBRARY_DIR}
      )
    SET(GLUT_glut_LIBRARY "-framework GLUT" CACHE STRING "GLUT library for OSX") 
    SET(GLUT_cocoa_LIBRARY "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
  ELSE (APPLE)
    
    FIND_PATH( GLUT_INCLUDE_DIR GL/glut.h
      /usr/include/GL
      /usr/openwin/share/include
      /usr/openwin/include
      /opt/graphics/OpenGL/include
      /opt/graphics/OpenGL/contrib/libglut
      )
  
    FIND_LIBRARY( GLUT_glut_LIBRARY glut
      /usr/openwin/lib
      )
    
    FIND_LIBRARY( GLUT_Xi_LIBRARY Xi
      /usr/openwin/lib
      )
    
    FIND_LIBRARY( GLUT_Xmu_LIBRARY Xmu
      /usr/openwin/lib
      )

    FIND_LIBRARY( GLUT_Xxf86vm_LIBRARY Xxf86vm
      /usr/openwin/lib
      )

  SET(GLUT_LIBRARY ${GLUT_glut_LIBRARY})
    
  ENDIF (APPLE)
  IF(GLUT_glut_LIBRARY)
    # Is -lXi and -lXmu required on all platforms that have it?
    # If not, we need some way to figure out what platform we are on.
    SET( GLUT_LIBRARIES
      ${GLUT_glut_LIBRARY}
      ${GLUT_Xmu_LIBRARY}
      ${GLUT_Xi_LIBRARY} 
      ${GLUT_Xxf86vm_LIBRARY} 
      ${GLUT_cocoa_LIBRARY}
      ) 
  ENDIF(GLUT_glut_LIBRARY)
  
ENDIF (WIN32)

IF(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLUT DEFAULT_MSG GLUT_LIBRARY_RELEASE GLUT_LIBRARY_DEBUG GLUT_INCLUDE_DIR)
  MARK_AS_ADVANCED(GLUT_LIBRARY_RELEASE GLUT_LIBRARY_DEBUG)
    SET( GLUT_LIBRARIES ${GLUT_LIBRARY}) 
ELSE(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLUT DEFAULT_MSG GLUT_LIBRARY GLUT_INCLUDE_DIR)
  MARK_AS_ADVANCED(GLUT_LIBRARY)
ENDIF(MSVC)

MARK_AS_ADVANCED(
  GLUT_INCLUDE_DIR
#  GLUT_glut_LIBRARY
#  GLUT_Xmu_LIBRARY
#  GLUT_Xi_LIBRARY
  )
