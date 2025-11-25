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

  IF (CMAKE_SIZEOF_VOID_P EQUAL 8)
    IF (FREEGLUT_LIBRARY)
      IF (NOT FREEGLUT_LIBRARY MATCHES "[/\\]x64[/\\]")
        UNSET(FREEGLUT_LIBRARY CACHE)
      ENDIF()
    ENDIF()
    IF (FREEGLUT_INCLUDE_DIR)
      UNSET(FREEGLUT_INCLUDE_DIR CACHE)
    ENDIF()
  ENDIF()

  FIND_PATH(FREEGLUT_INCLUDE_DIR
    NAMES GL/freeglut.h
    PATHS
      $ENV{EXTERNLIBS}/freeglut/include
      $ENV{EXTERNLIBS}/zebu/freeglut/include
      ${FREEGLUT_ROOT_PATH}/include
  )

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_FREEGLUT_LIB_PATHS
      $ENV{EXTERNLIBS}/freeglut/lib/x64
      $ENV{EXTERNLIBS}/zebu/freeglut/lib/x64
      ${FREEGLUT_ROOT_PATH}/lib/x64
    )
  else()
    set(_FREEGLUT_LIB_PATHS
      $ENV{EXTERNLIBS}/freeglut/lib
      $ENV{EXTERNLIBS}/zebu/freeglut/lib
      ${FREEGLUT_ROOT_PATH}/lib
    )
  endif()

  FIND_LIBRARY(FREEGLUT_LIBRARY
    NAMES freeglut freeglut_static
    PATHS ${_FREEGLUT_LIB_PATHS}
  )

  SET(FREEGLUT_LIBRARIES ${FREEGLUT_LIBRARY})
  MARK_AS_ADVANCED(FREEGLUT_LIBRARY)
  
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
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEGLUT DEFAULT_MSG
    FREEGLUT_LIBRARY FREEGLUT_INCLUDE_DIR)
  SET(FREEGLUT_LIBRARIES ${FREEGLUT_LIBRARY})
  MARK_AS_ADVANCED(FREEGLUT_LIBRARY)
ELSE(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(FREEGLUT DEFAULT_MSG
    FREEGLUT_LIBRARY FREEGLUT_INCLUDE_DIR)
  MARK_AS_ADVANCED(FREEGLUT_LIBRARY)
ENDIF(MSVC)

MARK_AS_ADVANCED(
  FREEGLUT_INCLUDE_DIR
#  FREEGLUT_glut_LIBRARY
#  FREEGLUT_Xmu_LIBRARY
#  FREEGLUT_Xi_LIBRARY
  )
