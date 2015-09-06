#
# Try to find VRPN library and include path.
# Once done this will define
#
# VRPN_FOUND
# VRPN_INCLUDE_DIR
# VRPN_LIBRARY
# 

find_path(VRPN_INCLUDE_DIR vrpn_Tracker.h
   PATHS
   $ENV{EXTERNLIBS}/VRPN/include
   /usr/local/include
   /usr/include
   /sw/include
   /opt/local/include
   DOC "The directory where vrpn_Tracker.h resides")

find_library(VRPN_VRPN_LIBRARY NAMES vrpns vrpn
   PATHS
   $ENV{EXTERNLIBS}/VRPN/lib
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The VRPN vrpn library")

find_library(VRPN_QUAT_LIBRARY NAMES quat
   PATHS
   $ENV{EXTERNLIBS}/VRPN/lib
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The VRPN quat library")

include(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(VRPN_VRPN_LIBRARY_DEBUG NAMES vrpnsD vrpnD
    PATHS
    $ENV{VRPN_HOME}/lib
    $ENV{EXTERNLIBS}/VRPN/lib
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local
    /usr
    /sw
    /opt/local
    /opt/csw
    /opt
    PATH_SUFFIXES lib64 lib libs64 libs libs/Win32 libs/Win64
    DOC "VRPN - Library (Debug)"
  )
  
find_library(VRPN_QUAT_LIBRARY_DEBUG NAMES quatD
   PATHS
   $ENV{EXTERNLIBS}/VRPN/lib
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The VRPN quat library (Debug)")
  
  IF(VRPN_LIBRARY_DEBUG AND VRPN_LIBRARY)
    SET(VRPN_LIBRARIES optimized ${VRPN_VRPN_LIBRARY} debug ${VRPN_VRPN_LIBRARY_DEBUG} optimized ${VRPN_QUAT_LIBRARY} debug ${VRPN_QUAT_LIBRARY_DEBUG})
  ENDIF(VRPN_LIBRARY_DEBUG AND VRPN_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(VRPN DEFAULT_MSG VRPN_LIBRARY VRPN_LIBRARY_DEBUG VRPN_INCLUDE_DIR)

  MARK_AS_ADVANCED(VRPN_VRPN_LIBRARY_DEBUG VRPN_VRPN_LIBRARY VRPN_QUAT_LIBRARY VRPN_QUAT_LIBRARY_DEBUG VRPN_INCLUDE_DIR)
  
ELSE(MSVC)

  set(VRPN_LIBRARIES ${VRPN_VRPN_LIBRARY} ${VRPN_QUAT_LIBRARY})
  
  find_package_handle_standard_args(VRPN DEFAULT_MSG VRPN_VRPN_LIBRARY VRPN_QUAT_LIBRARY VRPN_INCLUDE_DIR)
  
ENDIF(MSVC)

  if(VRPN_FOUND)
   set(VRPN_INCLUDE_DIRS ${VRPN_INCLUDE_DIR})
  endif()
