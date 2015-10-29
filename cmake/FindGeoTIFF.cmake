#
# Try to find GeoTIFF library and include path.
# Once done this will define
#
# GEOTIFF_FOUND
# GEOTIFF_INCLUDE_DIR
# GEOTIFF_LIBRARY
# 

find_path(GEOTIFF_INCLUDE_DIR geotiff.h
   PATHS
   /usr/local/include
   /usr/include
   /sw/include
   /usr/include/libgeotiff
   /opt/local/include
   DOC "The directory where geotiff.h resides")

find_library(GEOTIFF_LIBRARY NAMES geotiff
   PATHS
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The GeoTiff library")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GeoTIFF DEFAULT_MSG GEOTIFF_LIBRARY GEOTIFF_INCLUDE_DIR)

if(GEOTIFF_FOUND)
   set(GEOTIFF_INCLUDE_DIRS ${GEOTIFF_INCLUDE_DIR})
   set(GEOTIFF_LIBRARIES ${GEOTIFF_LIBRARY})
endif()
