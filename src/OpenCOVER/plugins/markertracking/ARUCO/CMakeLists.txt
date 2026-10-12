if(NOT OPENCV_FOUND)
    return()
endif()

if(NOT COVISE_USE_OPENCV4 AND NOT COVISE_USE_OPENCV5)
    message(WARNING
      "ARUCO plugin disabled: OpenCV not found or unsupported. "
      "Supported versions are OpenCV 4 and 5.")
    return()
endif()

SET(ARUCO_USE_CUDA FALSE)

IF(ARUCO_USE_CUDA)

IF(CUDA_FOUND)
# Example of how to find an include file from the CUDA_SDK_ROOT_DIR

find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")

mark_as_advanced(CUDA_CUT_INCLUDE_DIR)


# cutil library is called cutil64 for 64 bit builds on windows.  We don't want
# to get these confused, so we are setting the name based on the word size of
# the build.

IF(MSVC)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil64 libcutil_x86_64.a cutil libcutil.so)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil32)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

set(CUDA_SDK_SEARCH_PATH ${CUDA_SDK_SEARCH_PATH} $ENV{EXTERNLIBS}/cuda)
find_library(CUDA_CUT_LIBRARY
  NAMES cutil ${cuda_cutil_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib" "common/lib/x64" "lib" "C/lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
 )
# Now search system paths
find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutild_name cutil64D libcutil_x86_64.a cutil libcutil.so)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutild_name cutil32D)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

find_library(CUDA_CUT_LIBRARYD
  NAMES cutilD ${cuda_cutild_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib" "common/lib/x64" "lib" "C/lib"
  DOC "Location of cutil Debug library"
  NO_DEFAULT_PATH
 )
# Now search system paths
find_library(CUDA_CUT_LIBRARYD NAMES cutild ${cuda_cutild_name} DOC "Location of cutil library")

    SET(CUDA_CUT_LIBRARIES optimized ${CUDA_CUT_LIBRARY} debug ${CUDA_CUT_LIBRARYD})
ELSE(MSVC)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil64 libcutil_x86_64.a cutil libcutil.so)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil32)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

set(CUDA_SDK_SEARCH_PATH ${CUDA_SDK_SEARCH_PATH} $ENV{EXTERNLIBS}/cuda)
find_library(CUDA_CUT_LIBRARY
  NAMES cutil ${cuda_cutil_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib" "common/lib/x64" "lib" "C/lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
 )
# Now search system paths
find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
mark_as_advanced(CUDA_CUT_LIBRARY)
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
ENDIF(MSVC)
ENDIF(CUDA_FOUND)

ENDIF(ARUCO_USE_CUDA)


#INCLUDE_DIRECTORIES(${XERCESC_INCLUDE_DIR})

SET(HEADERS
  ARUCOTrackerPlugin.h
  Marker.h
  MatrixUtil.h
  ../common/ARCamera.h
#  CvCapture.h
  ../common/RemoteAR.h
)

SET(SOURCES
  ARUCOTrackerPlugin.cpp
  Marker.cpp
  MatrixUtil.cpp
  ../common/ARCamera.cpp
#  CvCapture.cpp
  ../common/RemoteAR.cpp
)


IF(ARUCO_USE_CUDA)
IF(CUDA_FOUND)
  cover_add_plugin(ARUCO ${CUDA_CUT_LIBRARIES} ${HEADERS} ${SOURCES})
ELSE(CUDA_FOUND)
  cover_add_plugin(ARUCO ${HEADERS} ${SOURCES})
ENDIF(CUDA_FOUND)
ELSE(ARUCO_USE_CUDA)
  cover_add_plugin(ARUCO ${HEADERS} ${SOURCES})
ENDIF(ARUCO_USE_CUDA)

if(TARGET ARUCO)
    target_link_libraries(ARUCO ${OPENCV_LIBRARIES})
    target_include_directories(ARUCO SYSTEM PRIVATE ${OPENCV_INCLUDE_DIRS})
endif()

SET(HEADERS
)
SET(SOURCES
  create_marker.cpp
)

ADD_COVISE_EXECUTABLE(createMarker)
COVISE_WNOERROR(createMarker)
TARGET_INCLUDE_DIRECTORIES(createMarker SYSTEM PRIVATE ${OPENCV_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(createMarker ${OPENCV_LIBRARIES})

