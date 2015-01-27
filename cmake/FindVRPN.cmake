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
   /usr/local/include
   /usr/include
   /sw/include
   /opt/local/include
   DOC "The directory where vrpn_Tracker.h resides")

find_library(VRPN_VRPN_LIBRARY NAMES vrpn
   PATHS
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The VRPN vrpn library")

find_library(VRPN_QUAT_LIBRARY NAMES quat
   PATHS
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The VRPN quat library")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VRPN DEFAULT_MSG VRPN_VRPN_LIBRARY VRPN_QUAT_LIBRARY VRPN_INCLUDE_DIR)

if(VRPN_FOUND)
   set(VRPN_INCLUDE_DIRS ${VRPN_INCLUDE_DIR})
   set(VRPN_LIBRARIES ${VRPN_VRPN_LIBRARY} ${VRPN_QUAT_LIBRARY})
endif()
