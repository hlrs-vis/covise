# Locate gdal
#
# This module accepts the following environment variables:
#
#    GDAL_DIR or GDAL_ROOT - Specify the location of GDAL
#
# This module defines the following CMake variables:
#
#    GDAL_FOUND - True if libgdal is found
#    GDAL_LIBRARY - A variable pointing to the GDAL library
#    GDAL_INCLUDE_DIR - Where to find the headers

#=============================================================================
# Copyright 2007-2009 Kitware, Inc.
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

#
# $GDALDIR is an environment variable that would
# correspond to the ./configure --prefix=$GDAL_DIR
# used in building gdal.
#
# Created by Eric Wing. I'm not a gdal user, but OpenSceneGraph uses it 
# for osgTerrain so I whipped this module together for completeness.
# I actually don't know the conventions or where files are typically
# placed in distros.
# Any real gdal users are encouraged to correct this (but please don't
# break the OS X framework stuff when doing so which is what usually seems 
# to happen).

# This makes the presumption that you are include gdal.h like
#
#include "gdal.h"

FIND_PATH(GDAL_INCLUDE_DIR gdal.h
  HINTS
    $ENV{GDAL_DIR}
    $ENV{GDAL_ROOT}
  PATH_SUFFIXES
     include/gdal
     include/GDAL
     include
  PATHS
      $ENV{EXTERNLIBS}/gdal
      ~/Library/Frameworks/gdal.framework/Headers
      /Library/Frameworks/gdal.framework/Headers
      /sw # Fink
      /opt/local # DarwinPorts
      /opt/csw # Blastwave
      /opt
)

IF(UNIX)
    # Use gdal-config to obtain the library version (this should hopefully
    # allow us to -lgdal1.x.y where x.y are correct version)
    # For some reason, libgdal development packages do not contain
    # libgdal.so...
    FIND_PROGRAM(GDAL_CONFIG gdal-config
        HINTS
          $ENV{GDAL_DIR}
          $ENV{GDAL_ROOT}
        PATH_SUFFIXES bin
        PATHS
            /sw # Fink
            /opt/local # DarwinPorts
            /opt/csw # Blastwave
            /opt
    )

    if(GDAL_CONFIG)
        exec_program(${GDAL_CONFIG} ARGS --libs OUTPUT_VARIABLE GDAL_CONFIG_LIBS)
        if(GDAL_CONFIG_LIBS)
            string(REGEX MATCHALL "-l[^ ]+" _gdal_dashl ${GDAL_CONFIG_LIBS})
            string(REGEX REPLACE "-l" "" _gdal_lib "${_gdal_dashl}")
            string(REGEX MATCHALL "-L[^ ]+" _gdal_dashL ${GDAL_CONFIG_LIBS})
            string(REGEX REPLACE "-L" "" _gdal_libpath "${_gdal_dashL}")
        endif()
    endif()
endif()

FIND_LIBRARY(GDAL_LIBRARY 
  NAMES ${_gdal_lib} gdal gdal_i gdal1.5.0 gdal1.4.0 gdal1.3.2 GDAL
  HINTS
     $ENV{GDAL_DIR}
     $ENV{GDAL_ROOT}
     ${_gdal_libpath}
  PATH_SUFFIXES lib64 lib
  PATHS
    $ENV{EXTERNLIBS}/gdal
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(GDAL_LIBRARY_DEBUG NAMES gdalD gdal_iD gdalD_i gdal1.5.0D gdal1.4.0D gdal1.3.2D
  HINTS
     $ENV{GDAL_DIR}
     $ENV{GDAL_ROOT}
     ${_gdal_libpath}
  PATH_SUFFIXES lib64 lib
  PATHS
    $ENV{EXTERNLIBS}/gdal
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
    DOC "GDAL - Library (Debug)"
  )
  
  IF(GDAL_LIBRARY_DEBUG AND GDAL_LIBRARY)
    SET(GDAL_LIBRARIES optimized ${GDAL_LIBRARY} debug ${GDAL_LIBRARY_DEBUG} general odbc32.lib general odbccp32.lib)
  ENDIF(GDAL_LIBRARY_DEBUG AND GDAL_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(GDAL DEFAULT_MSG GDAL_LIBRARY GDAL_LIBRARY_DEBUG GDAL_INCLUDE_DIR)

  MARK_AS_ADVANCED(GDAL_LIBRARY GDAL_LIBRARY_DEBUG GDAL_INCLUDE_DIR)
  
ELSE(MSVC)
  
   INCLUDE(FindPackageHandleStandardArgs)
   FIND_PACKAGE_HANDLE_STANDARD_ARGS(GDAL DEFAULT_MSG GDAL_LIBRARY GDAL_INCLUDE_DIR)

  MARK_AS_ADVANCED(GDAL_LIBRARY GDAL_INCLUDE_DIR)
   set(GDAL_LIBRARIES ${GDAL_LIBRARY})
  
ENDIF(MSVC)

set(GDAL_INCLUDE_DIRS ${GDAL_INCLUDE_DIR})
