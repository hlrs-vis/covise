#
# Try to find IceT compositing library and include path.
# Once done this will define
#
# ICET_FOUND
# ICET_INCLUDE_DIR
# ICET_LIBRARY
# 

find_path(ICET_INCLUDE_DIR IceT.h
   PATHS
   /usr/local/include
   /usr/include
   /sw/include
   /opt/local/include
   DOC "The directory where IceT.h resides")

find_library(ICET_CORE_LIBRARY NAMES IceTCore
   PATHS
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The IceTCore library")

find_library(ICET_GL_LIBRARY NAMES IceTGL
   PATHS
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The IceTGL library")

find_library(ICET_MPI_LIBRARY NAMES IceTMPI
   PATHS
   /usr/local
   /usr
   /sw
   /opt/local
   PATH_SUFFIXES lib lib64
   DOC "The IceTMPI library")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ICET DEFAULT_MSG ICET_CORE_LIBRARY ICET_GL_LIBRARY ICET_MPI_LIBRARY ICET_INCLUDE_DIR)

if(ICET_FOUND)
   set(ICET_INCLUDE_DIRS ${ICET_INCLUDE_DIR})
   set(ICET_LIBRARIES ${ICET_CORE_LIBRARY} ${ICET_GL_LIBRARY} ${ICET_MPI_LIBRARY})
endif()
