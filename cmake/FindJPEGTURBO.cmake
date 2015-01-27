# - Find JPEGTURBO
# Find the libjpeg-turbo includes and library
# This module defines
#  JPEGTURBO_INCLUDE_DIR, where to find jpeglib.h and turbojpeg.h, etc.
#  JPEGTURBO_LIBRARIES, the libraries needed to use libjpeg-turbo.
#  JPEGTURBO_FOUND, If false, do not try to use libjpeg-turbo.
# also defined, but not for general use are
#  JPEGTURBO_LIBRARY, where to find the libjpeg-turbo library.

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

#FIND_PATH(JPEGTURBO_INCLUDE_DIR turbojpeg.h)

find_path(JPEGTURBO_PREFIX "include/turbojpeg.h"
  $ENV{JPEGTURBO_HOME}
  $ENV{EXTERNLIBS}/libjpeg-turbo64
  $ENV{EXTERNLIBS}/libjpeg-turbo
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr/local/opt/jpeg-turbo # Homebrew
  /usr
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt
  DOC "JPEGTURBO - Prefix"
)


FIND_PATH(JPEGTURBO_INCLUDE_DIR "turbojpeg.h"
  HINTS ${JPEGTURBO_PREFIX}/include
  PATHS
  $ENV{JPEGTURBO_HOME}/include
  $ENV{EXTERNLIBS}/libjpeg-turbo64/include
  $ENV{EXTERNLIBS}/libjpeg-turbo/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/local/opt/jpeg-turbo/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "JPEGTURBO - Headers"
)

FIND_PATH(JPEGTURBO_INCLUDE_DIR_INT "jpegint.h"
   PATHS ${JPEGTURBO_INCLUDE_DIR}
   DOC "JPEGTURBO - Internal Headers"
)

#FIND_LIBRARY(TURBOJPEG_LIBRARY NAMES turbojpeg)
#FIND_LIBRARY(JPEGTURBO_LIBRARY NAMES jpeg)

FIND_LIBRARY(JPEGTURBO_LIBRARY NAMES jpeg
  HINTS ${JPEGTURBO_PREFIX}/lib ${JPEGTURBO_PREFIX}/lib64
  PATHS
  $ENV{JPEGTURBO_HOME}
  $ENV{EXTERNLIBS}/libjpeg-turbo64
  $ENV{EXTERNLIBS}/libjpeg-turbo
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr/local/opt/jpeg-turbo
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "JPEGTURBO - Library"
)

FIND_LIBRARY(TURBOJPEG_LIBRARY turbojpeg
  HINTS ${JPEGTURBO_PREFIX}/lib ${JPEGTURBO_PREFIX}/lib64
  PATHS
  $ENV{JPEGTURBO_HOME}
  $ENV{EXTERNLIBS}/libjpeg-turbo64
  $ENV{EXTERNLIBS}/libjpeg-turbo
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr/local/opt/jpeg-turbo
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "JPEGTURBO - Library"
)

FIND_LIBRARY(TURBOJPEG_LIBRARY_STATIC NAMES libturbojpeg.a
  HINTS ${JPEGTURBO_PREFIX}/lib ${JPEGTURBO_PREFIX}/lib64
  PATHS
  $ENV{JPEGTURBO_HOME}
  $ENV{EXTERNLIBS}/libjpeg-turbo64
  $ENV{EXTERNLIBS}/libjpeg-turbo
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr/local/opt/jpeg-turbo
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "JPEGTURBO - Static library"
)

# handle the QUIETLY and REQUIRED arguments and set JPEGTURBO_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(JPEGTURBO DEFAULT_MSG TURBOJPEG_LIBRARY JPEGTURBO_LIBRARY JPEGTURBO_INCLUDE_DIR)

IF(JPEGTURBO_FOUND)
  SET(JPEGTURBO_LIBRARIES ${JPEGTURBO_LIBRARY})
  if (TURBOJPEG_LIBRARY_STATIC)
     SET(TURBOJPEG_LIBRARIES ${TURBOJPEG_LIBRARY_STATIC})
  else()
     SET(TURBOJPEG_LIBRARIES ${TURBOJPEG_LIBRARY})
  endif()

  INCLUDE (CheckSymbolExists)
  set(CMAKE_REQUIRED_INCLUDES ${JPEGTURBO_INCLUDE_DIR})
  CHECK_SYMBOL_EXISTS(tjMCUWidth "turbojpeg.h" TURBOJPEG_HAVE_TJMCUWIDTH)

  if (JPEGTURBO_INCLUDE_DIR_INT)
     set(TURBOJPEG_HAVE_INTERNAL TRUE)
  else()
     set(TURBOJPEG_HAVE_INTERNAL FALSE)
  endif()
ENDIF(JPEGTURBO_FOUND)

MARK_AS_ADVANCED(TURBOJPEG_LIBRARY JPEGTURBO_LIBRARY JPEGTURBO_INCLUDE_DIR TURBOJPEG_LIBRARY_STATIC TURBOJPEG_HAVE_TJMCUWIDTH TURBOJPEG_HAVE_INTERNAL)
