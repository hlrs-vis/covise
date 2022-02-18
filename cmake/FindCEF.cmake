# Copyright (c) 2016 The Chromium Embedded Framework Authors. All rights
# reserved. Use of this source code is governed by a BSD-style license that
# can be found in the LICENSE file.

#
# This file is the CEF CMake configuration entry point and should be loaded
# using `find_package(CEF REQUIRED)`. See the top-level CMakeLists.txt file
# included with the CEF binary distribution for usage information.
#

# - Find CEF
# Find the CEF includes and library
#
#  CEF_INCLUDE_DIR - Where to find CEF includes
#  CEF_LIBRARIES   - List of libraries when using CEF
#  CEF_FOUND       - True if CEF was found

IF(CEF_INCLUDE_DIR)
  SET(CEF_FIND_QUIETLY TRUE)
ENDIF(CEF_INCLUDE_DIR)

FIND_PATH(CEF_INCLUDE_DIR "include/base/cef_macros.h"
  PATHS
  $ENV{CEF_HOME}/include
  $ENV{EXTERNLIBS}/cef/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  PATH_SUFFIXES CEF
  DOC "CEF - Headers"
)

SET(CEF_NAMES cef libcef "Chromium Embedded Framework")
SET(CEF_DBG_NAMES cefd libcefd)

FIND_LIBRARY(CEF_LIBRARY NAMES ${CEF_NAMES}
  PATHS
  $ENV{CEF_HOME}
  $ENV{EXTERNLIBS}/cef
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES Release lib/CEF lib64/CEF lib lib64 lib/win_64_VS2015
  DOC "CEF - Library"
)
SET(CEF_WRAPPER_NAMES libcef_dll_wrapper.a libcef_dll_wrapper.lib)
SET(CEF_WRAPPER_DBG_NAMES libcef_dll_wrapperD.a libcef_dll_wrapperD.lib)

FIND_LIBRARY(CEF_WRAPPER_LIBRARY NAMES ${CEF_WRAPPER_NAMES}
  PATHS
  $ENV{CEF_HOME}
  $ENV{EXTERNLIBS}/cef
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES Debug lib/CEF lib64/CEF lib lib64 lib/win_64_VS2015
  DOC "CEF - Library"
)


INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(CEF_LIBRARY_DEBUG NAMES ${CEF_DBG_NAMES}
    PATHS
    $ENV{CEF_HOME}/lib
    $ENV{EXTERNLIBS}/cef
    PATH_SUFFIXES lib lib64 lib/win_64_VS2015
    DOC "CEF - Library (Debug)"
  )
  FIND_LIBRARY(CEF_WRAPPER_LIBRARY_DEBUG NAMES ${CEF_WRAPPER_DBG_NAMES}
    PATHS
    $ENV{CEF_HOME}/lib
    $ENV{EXTERNLIBS}/cef
    PATH_SUFFIXES lib lib64 lib/win_64_VS2015
    DOC "CEF - _WRAPPER Library (Debug)"
  )
  
  
  IF(CEF_LIBRARY_DEBUG AND CEF_LIBRARY)
    SET(CEF_LIBRARIES optimized ${CEF_LIBRARY} debug ${CEF_LIBRARY_DEBUG} optimized ${CEF_WRAPPER_LIBRARY} debug ${CEF_WRAPPER_LIBRARY_DEBUG} )
  ENDIF(CEF_LIBRARY_DEBUG AND CEF_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(CEF DEFAULT_MSG CEF_LIBRARY CEF_LIBRARY_DEBUG CEF_INCLUDE_DIR)

  MARK_AS_ADVANCED(CEF_LIBRARY CEF_LIBRARY_DEBUG CEF_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  SET(CEF_LIBRARIES ${CEF_LIBRARY} ${CEF_WRAPPER_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(CEF DEFAULT_MSG CEF_LIBRARY CEF_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(CEF_LIBRARY CEF_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(CEF_FOUND)
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
MESSAGE("CEF_FOUND")
  SET(CEF_INCLUDE_DIRS ${CEF_INCLUDE_DIR})
ELSE(CEF_FOUND)
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")
MESSAGE("NOT CEF_FOUND")




# Find the CEF binary distribution root directory.
set(_CEF_ROOT "")
if(CEF_ROOT AND IS_DIRECTORY "${CEF_ROOT}")
  set(_CEF_ROOT "${CEF_ROOT}")
  set(_CEF_ROOT_EXPLICIT 1)
else()
  set(_ENV_CEF_ROOT "")
  if(DEFINED ENV{CEF_ROOT})
    file(TO_CMAKE_PATH "$ENV{CEF_ROOT}" _ENV_CEF_ROOT)
  endif()
  if(_ENV_CEF_ROOT AND IS_DIRECTORY "${_ENV_CEF_ROOT}")
    set(_CEF_ROOT "${_ENV_CEF_ROOT}")
    set(_CEF_ROOT_EXPLICIT 1)
  endif()
  unset(_ENV_CEF_ROOT)
endif()

set(ERROR FASLE)
if(NOT DEFINED _CEF_ROOT_EXPLICIT)
  message(WARNING "Specify a CEF_ROOT value via CMake or environment variable to find CEF")
  set(ERROR TRUE)

endif()

if(NOT IS_DIRECTORY "${_CEF_ROOT}/cmake")
  message(WARNING "No CMake bootstrap found for CEF binary distribution at: ${CEF_ROOT}.")
  set(ERROR TRUE)
endif()

if(NOT ERROR)
  # Execute additional cmake files from the CEF binary distribution.
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${_CEF_ROOT}/cmake")
  include("cef_variables")
  include("cef_macros")
endif()
ENDIF(CEF_FOUND)
