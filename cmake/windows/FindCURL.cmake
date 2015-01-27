# - Find curl
# Find the native CURL headers and libraries.
#
#  CURL_INCLUDE_DIRS - where to find curl/curl.h, etc.
#  CURL_LIBRARIES    - List of libraries when using curl.
#  CURL_FOUND        - True if curl found.

#=============================================================================
# Copyright 2006-2009 Kitware, Inc.
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

IF(CURL_INCLUDE_DIR)
  SET(CURL_FIND_QUIETLY TRUE)
ENDIF(CURL_INCLUDE_DIR)


# Look for the header file.

FIND_PATH(CURL_INCLUDE_DIR curl/curl.h
   PATHS
   $ENV{CURL_HOME}/include
   $ENV{EXTERNLIBS}/curl/include
   DOC "curl - Headers"
   NO_DEFAULT_PATH
)
MARK_AS_ADVANCED(CURL_INCLUDE_DIR)

IF (MSVC)
    # check whether this is a /MT(d) build
    STRING(REGEX MATCH "[mM][tT][dD]" MTD_COMPILE_OPTION ${CMAKE_C_FLAGS_DEBUG})
    IF (MTD_COMPILE_OPTION)
      # MESSAGE("Using static MS-Runtime !!!")
      FIND_LIBRARY(CURL_LIBRARY_DEBUG NAMES curld libcurld curllibd libcurld_imp curllibd_static
        PATHS
        $ENV{CURL_HOME}/lib
        $ENV{EXTERNLIBS}/CURL/lib
      )
      FIND_LIBRARY(CURL_LIBRARY_RELEASE NAMES curl libcurl curllib libcurl_imp curllib_static
        PATHS
        $ENV{CURL_HOME}/lib
        $ENV{EXTERNLIBS}/CURL/lib
      )
    ELSE (MTD_COMPILE_OPTION)
      FIND_LIBRARY(CURL_LIBRARY_DEBUG NAMES curld libcurld curllibd libcurld_imp curllibd_static
        PATHS
        $ENV{CURL_HOME}/lib
        $ENV{EXTERNLIBS}/CURL/lib
      )
      FIND_LIBRARY(CURL_LIBRARY_RELEASE NAMES curl libcurl curllib libcurl_imp curllib_static
        PATHS
        $ENV{CURL_HOME}/lib
        $ENV{EXTERNLIBS}/CURL/lib
      )
    ENDIF (MTD_COMPILE_OPTION)

    IF(MSVC_IDE)
      IF (CURL_LIBRARY_DEBUG AND CURL_LIBRARY_RELEASE)
         SET(CURL_LIBRARIES optimized ${CURL_LIBRARY_RELEASE} debug ${CURL_LIBRARY_DEBUG})
      ELSE (CURL_LIBRARY_DEBUG AND CURL_LIBRARY_RELEASE)
         SET(CURL_LIBRARIES NOTFOUND)
         MESSAGE(STATUS "Could not find the debug AND release version of CURL")
      ENDIF (CURL_LIBRARY_DEBUG AND CURL_LIBRARY_RELEASE)
    ELSE(MSVC_IDE)
      STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_TOLOWER)
      IF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(CURL_LIBRARIES ${CURL_LIBRARY_DEBUG})
      ELSE(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(CURL_LIBRARIES ${CURL_LIBRARY_RELEASE})
      ENDIF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
    ENDIF(MSVC_IDE)
    MARK_AS_ADVANCED(CURL_LIBRARY_DEBUG CURL_LIBRARY_RELEASE)

ELSE (MSVC)
  
  SET(CURL_NAMES ${CURL_NAMES} curl libcurl curllib libcurl_imp curllib_static)
  FIND_LIBRARY(CURL_LIBRARY NAMES ${CURL_NAMES}
    PATHS
    $ENV{CURL_HOME}/lib
    $ENV{EXTERNLIBS}/CURL/lib
    NO_DEFAULT_PATH
  )
  
  IF (CURL_LIBRARY)
    SET(CURL_LIBRARIES ${CURL_LIBRARY})
  ELSE (CURL_LIBRARY)
    SET(CURL_LIBRARIES NOTFOUND)
    MESSAGE(STATUS "Could not find CURL-library")    
  ENDIF (CURL_LIBRARY)
  MARK_AS_ADVANCED(CURL_LIBRARY)
  
ENDIF (MSVC)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(CURL DEFAULT_MSG CURL_LIBRARY_RELEASE CURL_LIBRARY_DEBUG CURL_INCLUDE_DIR)
  MARK_AS_ADVANCED(CURL_LIBRARY_RELEASE CURL_LIBRARY_DEBUG)
ELSE(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(CURL DEFAULT_MSG CURL_LIBRARY CURL_INCLUDE_DIR)
  MARK_AS_ADVANCED(CURL_LIBRARY)
ENDIF(MSVC)


IF(CURL_FOUND)
  SET(CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIR})
ENDIF(CURL_FOUND)
