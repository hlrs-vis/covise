# - try to find glut library and include files
#  FREEGLUT_INCLUDE_DIR, where to find GL/glut.h, etc.
#  FREEGLUT_LIBRARIES, the libraries to link against
#  FREEGLUT_FOUND, If false, do not try to use FREEGLUT.
# Also defined, but not for general use are:
#  FREEGLUT_glut_LIBRARY = the full path to the glut library.
#  FREEGLUT_Xmu_LIBRARY  = the full path to the Xmu library.
#  FREEGLUT_Xi_LIBRARY   = the full path to the Xi Library.

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

  SET(FREEGLUT_PACKAGE_DEFINITIONS "-DFREEGLUT_NO_LIB_PRAGMA")
  FIND_PATH( FREEGLUT_INCLUDE_DIR NAMES GL/glut.h 
    PATHS  
    $ENV{EXTERNLIBS}/freeglut/include
    ${FREEGLUT_ROOT_PATH}/include )
  FIND_LIBRARY( FREEGLUT_LIBRARY_DEBUG NAMES glutD glut32D freeglutD glut freeglut
    PATHS
    $ENV{EXTERNLIBS}/freeglut/lib
    ${OPENGL_LIBRARY_DIR}
    ${FREEGLUT_ROOT_PATH}/Debug
    )
   if(NOT FREEGLUT_LIBRARY_DEBUG)
       FIND_LIBRARY( FREEGLUT_LIBRARY_DEBUG NAMES glut glut32 freeglut
           PATHS $ENV{EXTERNLIBS}/freeglut/lib
           ${OPENGL_LIBRARY_DIR}/debug/lib
           ${FREEGLUT_ROOT_PATH}/debug/lib
           )
   endif()
    FIND_LIBRARY( FREEGLUT_LIBRARY_RELEASE NAMES glut glut32 freeglut
    PATHS
    ${OPENGL_LIBRARY_DIR}
    $ENV{EXTERNLIBS}/freeglut/lib
    ${FREEGLUT_ROOT_PATH}/Release
    )
    IF(CMAKE_CONFIGURATION_TYPES)
      IF (FREEGLUT_LIBRARY_DEBUG AND FREEGLUT_LIBRARY_RELEASE)
         SET(FREEGLUT_LIBRARY optimized ${FREEGLUT_LIBRARY_RELEASE} debug ${FREEGLUT_LIBRARY_DEBUG})
      ELSE (FREEGLUT_LIBRARY_DEBUG AND FREEGLUT_LIBRARY_RELEASE)
         SET(FREEGLUT_LIBRARY NOTFOUND)
         MESSAGE(STATUS "Could not find the debug AND release version of zlib")
      ENDIF (FREEGLUT_LIBRARY_DEBUG AND FREEGLUT_LIBRARY_RELEASE)
    ELSE()
      STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_TOLOWER)
      IF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(FREEGLUT_LIBRARY ${FREEGLUT_LIBRARY_DEBUG})
      ELSE(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(FREEGLUT_LIBRARY ${FREEGLUT_LIBRARY_RELEASE})
      ENDIF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
    ENDIF()
    MARK_AS_ADVANCED(FREEGLUT_LIBRARY_DEBUG FREEGLUT_LIBRARY_RELEASE)
  ELSE (WIN32)
  
  IF (APPLE)
    # These values for Apple could probably do with improvement.
    FIND_PATH( FREEGLUT_INCLUDE_DIR glut.h
      /System/Library/Frameworks/FREEGLUT.framework/Versions/A/Headers
      ${OPENGL_LIBRARY_DIR}
      )
    SET(FREEGLUT_glut_LIBRARY "-framework FREEGLUT" CACHE STRING "FREEGLUT library for OSX") 
    SET(FREEGLUT_cocoa_LIBRARY "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
  ELSE (APPLE)
    
    FIND_PATH( FREEGLUT_INCLUDE_DIR GL/glut.h
      /usr/include/GL
      /usr/openwin/share/include
      /usr/openwin/include
      /opt/graphics/OpenGL/include
      /opt/graphics/OpenGL/contrib/libglut
      )
  
    FIND_LIBRARY( FREEGLUT_glut_LIBRARY glut
      /usr/openwin/lib
      )
    
    FIND_LIBRARY( FREEGLUT_Xi_LIBRARY Xi
      /usr/openwin/lib
      )
    
    FIND_LIBRARY( FREEGLUT_Xmu_LIBRARY Xmu
      /usr/openwin/lib
      )

    FIND_LIBRARY( FREEGLUT_Xxf86vm_LIBRARY Xxf86vm
      /usr/openwin/lib
      )

  SET(FREEGLUT_LIBRARY ${FREEGLUT_glut_LIBRARY})
    
  ENDIF (APPLE)
  IF(FREEGLUT_glut_LIBRARY)
    # Is -lXi and -lXmu required on all platforms that have it?
    # If not, we need some way to figure out what platform we are on.
    SET( FREEGLUT_LIBRARIES
      ${FREEGLUT_glut_LIBRARY}
      ${FREEGLUT_Xmu_LIBRARY}
      ${FREEGLUT_Xi_LIBRARY} 
      ${FREEGLUT_Xxf86vm_LIBRARY} 
      ${FREEGLUT_cocoa_LIBRARY}
      ) 
  ENDIF(FREEGLUT_glut_LIBRARY)
  
ENDIF (WIN32)

IF(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEGLUT DEFAULT_MSG FREEGLUT_LIBRARY_RELEASE FREEGLUT_LIBRARY_DEBUG FREEGLUT_INCLUDE_DIR)
  MARK_AS_ADVANCED(FREEGLUT_LIBRARY_RELEASE FREEGLUT_LIBRARY_DEBUG)
    SET( FREEGLUT_LIBRARIES ${FREEGLUT_LIBRARY}) 
ELSE(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEGLUT DEFAULT_MSG FREEGLUT_LIBRARY FREEGLUT_INCLUDE_DIR)
  MARK_AS_ADVANCED(FREEGLUT_LIBRARY)
ENDIF(MSVC)

MARK_AS_ADVANCED(
  FREEGLUT_INCLUDE_DIR
#  FREEGLUT_glut_LIBRARY
#  FREEGLUT_Xmu_LIBRARY
#  FREEGLUT_Xi_LIBRARY
  )
