# - Try to find wmfsdk11-library
# Once done this will define
#
#  WMFSDK_INCLUDE_DIR    - where to find wmfsdk11.h, etc.
#  WMFSDK_INCLUDE_DIRS   - same as above (uncached version)
#  WMFSDK_LIBRARIES      - list of libraries when using wmfsdk11.
#  WMFSDK_FOUND          - True if wmfsdk11 was found.

IF(WMFSDK_INCLUDE_DIR)
  SET(WMFSDK_FIND_QUIETLY TRUE)
ENDIF(WMFSDK_INCLUDE_DIR)

FIND_PATH(WMFSDK_INCLUDE_DIR wmsdk.h
   PATHS
   $ENV{WMFSDK_HOME}/include
   $ENV{WindowsSdkDir}/Include
   $ENV{WindowsSdkDir}/Include/um
   $ENV{EXTERNLIBS}/wmfsdk11/include
   "/Program Files/Microsoft SDKs/Windows/v7.0/Include"
   "/Program Files (x86)/Microsoft SDKs/Windows/v7.1A/Include"
   DOC "wmfsdk11 - Headers"
   NO_DEFAULT_PATH
)

FIND_PATH(WMFSDK_SHARED_INCLUDE_DIR winapifamily.h
   PATHS
   $ENV{WMFSDK_HOME}/include
   $ENV{WindowsSdkDir}/Include
   $ENV{WindowsSdkDir}/Include/shared
   $ENV{EXTERNLIBS}/wmfsdk11/include
   "/Program Files/Microsoft SDKs/Windows/v7.0/Include"
   "/Program Files (x86)/Microsoft SDKs/Windows/v7.1A/Include"
   DOC "shared - Headers"
   NO_DEFAULT_PATH
)

FIND_PATH(WMFSDK_INCLUDE_DIR wmfsdk11.h DOC "wmfsdk11 - Headers")

MARK_AS_ADVANCED(WMFSDK_INCLUDE_DIR)


IF (MSVC)
    
  FIND_LIBRARY(WMFSDK_LIBRARY NAMES wmvcore
    PATHS
    $ENV{WMFSDK_HOME}/lib
   $ENV{EXTERNLIBS}/wmfsdk11/lib/x64
   $ENV{EXTERNLIBS}/wmfsdk11/lib
   $ENV{WindowsSdkDir}/Lib/x64
   $ENV{WindowsSdkDir}/Lib/win8/um/x64
   $ENV{WindowsSdkDir}/Lib/winv6.3/um/x64
   "/Program Files/Microsoft SDKs/Windows/v7.0/Lib/x64"
   "/Program Files (x86)/Microsoft SDKs/Windows/v7.1A/Lib/x64"
    NO_DEFAULT_PATH
  )
  IF (WMFSDK_LIBRARY)
    SET(WMFSDK_LIBRARIES ${WMFSDK_LIBRARY})
  ELSE (WMFSDK_LIBRARY)
    SET(WMFSDK_LIBRARIES NOTFOUND)
    MESSAGE(STATUS "Could not find wmfsdk11-library")    
  ENDIF (WMFSDK_LIBRARY)
  MARK_AS_ADVANCED(WMFSDK_LIBRARY)
  
ENDIF (MSVC)

INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(WMFSDK DEFAULT_MSG WMFSDK_LIBRARIES WMFSDK_INCLUDE_DIR)
  MARK_AS_ADVANCED(WMFSDK_LIBRARIES)
ELSE(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(WMFSDK DEFAULT_MSG WMFSDK_LIBRARY WMFSDK_INCLUDE_DIR)
  MARK_AS_ADVANCED(WMFSDK_LIBRARY)
ENDIF(MSVC)
