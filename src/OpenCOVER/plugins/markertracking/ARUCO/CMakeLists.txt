if(NOT COVISE_USE_OPENCV4 AND NOT COVISE_USE_OPENCV5)
    message(WARNING
      "ARUCO plugin disabled: OpenCV not found or unsupported. "
      "Supported versions are OpenCV 4 and 5.")
    return()
endif()
USING(OpenCV)

# calib: OpenCV4=calib3d, some OpenCV5 packages=calib
set(ARUCO_CALIB_LIB "${OPENCV_CALIB3D_LIBRARY}")
if((NOT ARUCO_CALIB_LIB OR ARUCO_CALIB_LIB STREQUAL "OPENCV_CALIB3D_LIBRARY-NOTFOUND")
   AND DEFINED OPENCV_CALIB_LIBRARY
   AND NOT OPENCV_CALIB_LIBRARY STREQUAL "OPENCV_CALIB_LIBRARY-NOTFOUND")
    set(ARUCO_CALIB_LIB "${OPENCV_CALIB_LIBRARY}")
endif()

# geometry: needed on some OpenCV5 package layouts (solvePnP/Rodrigues)
set(ARUCO_GEOMETRY_LIB "")
if(COVISE_USE_OPENCV5)
    unset(ARUCO_GEOMETRY_LIB CACHE)
    get_filename_component(_opencv_libdir "${OPENCV_CORE_LIBRARY}" DIRECTORY)

    # 1) direct known path fallback
    if(EXISTS "${_opencv_libdir}/libopencv_geometry.so")
        set(ARUCO_GEOMETRY_LIB "${_opencv_libdir}/libopencv_geometry.so")
    else()
        # 2) regular lookup
        find_library(ARUCO_GEOMETRY_LIB
            NAMES opencv_geometry
            HINTS "${_opencv_libdir}"
            PATHS /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64
        )
    endif()

    if(NOT ARUCO_GEOMETRY_LIB OR ARUCO_GEOMETRY_LIB STREQUAL "ARUCO_GEOMETRY_LIB-NOTFOUND")
        message(FATAL_ERROR "ARUCO: OpenCV5 needs opencv_geometry for solvePnP/Rodrigues")
    endif()
endif()

set(ARUCO_OPENCV_LIBS
    ${OPENCV_CORE_LIBRARY}
    ${OPENCV_IMGPROC_LIBRARY}
    ${OPENCV_VIDEOIO_LIBRARY}
    ${OPENCV_HIGHGUI_LIBRARY}
    ${OPENCV_OBJDETECT_LIBRARY}
    ${ARUCO_CALIB_LIB}
    ${ARUCO_GEOMETRY_LIB}
)
list(FILTER ARUCO_OPENCV_LIBS EXCLUDE REGEX ".*-NOTFOUND$")
list(REMOVE_ITEM ARUCO_OPENCV_LIBS "")
list(REMOVE_DUPLICATES ARUCO_OPENCV_LIBS)

message(STATUS "ARUCO final OpenCV libs: ${ARUCO_OPENCV_LIBS}")

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
    target_link_libraries(ARUCO ${ARUCO_OPENCV_LIBS})
endif()

SET(HEADERS
)
SET(SOURCES
  create_marker.cpp
)

ADD_COVISE_EXECUTABLE(createMarker)
COVISE_WNOERROR(createMarker)

TARGET_LINK_LIBRARIES(createMarker ${EXTRA_LIBS} )

