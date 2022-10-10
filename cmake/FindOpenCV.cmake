# - Find OPENCV
# Find the OPENCV includes and library
#
#  OPENCV_INCLUDE_DIR - Where to find OPENCV includes
#  OPENCV_LIBRARIES   - List of libraries when using OPENCV
#  OPENCV_FOUND       - True if OPENCV was found

IF(OPENCV_INCLUDE_DIR)
  SET(OPENCV_FIND_QUIETLY TRUE)
ENDIF(OPENCV_INCLUDE_DIR)

IF(COVISE_USE_OPENCV4)
    set(MAJOR 4)
    set(SUFFIXES opencv4)
    SET(OPENCV_EXTERNLIBS $ENV{EXTERNLIBS}/opencv4)
ELSEIF(COVISE_USE_OPENCV3)
    set(MAJOR 3)
    set(SUFFIXES opencv3)
    SET(OPENCV_EXTERNLIBS $ENV{EXTERNLIBS}/opencv3)
ELSE()
    SET(OPENCV_EXTERNLIBS $ENV{EXTERNLIBS}/OpenCV/build)
    set(SUFFIXES opencv4 opencv3)
ENDIF()


FIND_PATH(OPENCV_INCLUDE_DIR "opencv2/core/core.hpp"
  PATHS
  ${OPENCV_EXTERNLIBS}/include
  $ENV{OPENCV_HOME}/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  /opt/homebrew/opt/opencv/include
  /opt/homebrew/opt/opencv@${MAJOR}/include
  /usr/local/opt/opencv/include
  /usr/local/opt/opencv@${MAJOR}/include
  DOC "OpenCV - Headers"
  PATH_SUFFIXES ${SUFFIXES}
)

INCLUDE(FindPackageHandleStandardArgs)

MACRO(FIND_OPENCV_COMPONENT component version)

STRING(TOUPPER ${component} _uppercomponent)
SET(OPENCV_NAMES opencv_${component}${version} opencv_${component})
SET(OPENCV_WOV_NAMES opencv_${component})
SET(OPENCV_DBG_NAMES opencv_${component}${version}d)

FIND_LIBRARY(OPENCV_${_uppercomponent}_LIBRARY NAMES ${OPENCV_NAMES}
  PATHS
  ${OPENCV_EXTERNLIBS}
  $ENV{OPENCV_HOME}
  ${OPENCV_EXTERNLIBS}/x64/vc17
  ${OPENCV_EXTERNLIBS}/x64/vc14
  ${OPENCV_EXTERNLIBS}/x64/vc12
  ${OPENCV_EXTERNLIBS}/x64/vc11
  ${OPENCV_EXTERNLIBS}/x64/vc10
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  /opt/homebrew/opt/opencv
  /opt/homebrew/opt/opencv@${MAJOR}
  /usr/local/opt/opencv
  /usr/local/opt/opencv@${MAJOR}
  PATH_SUFFIXES lib lib64
  DOC "OPENCV - Library"
)
FIND_LIBRARY(OPENCV_${_uppercomponent}_LIBRARY NAMES ${OPENCV_NAMES}
  DOC "OPENCV - Library"
)

IF(MSVC)
  # VisualStudio needs a debug version
  #MESSAGE(${OPENCV_DBG_NAMES})
  FIND_LIBRARY(OPENCV_${_uppercomponent}_LIBRARY_DEBUG NAMES ${OPENCV_DBG_NAMES}
    PATHS
    $ENV{OPENCV_HOME}/lib
    ${OPENCV_EXTERNLIBS}/x64/vc17/lib
    ${OPENCV_EXTERNLIBS}/x64/vc14/lib
    ${OPENCV_EXTERNLIBS}/x64/vc12/lib
    ${OPENCV_EXTERNLIBS}/x64/vc11/lib
    ${OPENCV_EXTERNLIBS}/x64/vc10/lib
    PATH_SUFFIXES debug/lib debug/lib64 lib lib64
    DOC "OPENCV - Library (Debug)"
  )
  
  IF(OPENCV_${_uppercomponent}_LIBRARY_DEBUG AND OPENCV_${_uppercomponent}_LIBRARY)
    SET(OPENCV_LIBRARIES ${OPENCV_LIBRARIES} optimized ${OPENCV_${_uppercomponent}_LIBRARY} debug ${OPENCV_${_uppercomponent}_LIBRARY_DEBUG})
  ENDIF(OPENCV_${_uppercomponent}_LIBRARY_DEBUG AND OPENCV_${_uppercomponent}_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCV_${_uppercomponent} REQUIRED_VARS OPENCV_${_uppercomponent}_LIBRARY OPENCV_${_uppercomponent}_LIBRARY_DEBUG OPENCV_INCLUDE_DIR NAME_MISMATCHED)

  MARK_AS_ADVANCED(OPENCV_${_uppercomponent}_LIBRARY OPENCV_${_uppercomponent}_LIBRARY_DEBUG OPENCV_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world

  FIND_LIBRARY(OPENCV_${_uppercomponent}_LIBRARY NAMES ${OPENCV_WOV_NAMES}
    PATHS
    $ENV{OPENCV_HOME}/lib
    DOC "OPENCV - Library (WOV)"
  )

  SET(OPENCV_LIBRARIES ${OPENCV_LIBRARIES} ${OPENCV_${_uppercomponent}_LIBRARY})
  if (APPLE AND ${_uppercomponent} STREQUAL "ARUCO")
      SET(OPENCV_LIBRARIES ${OPENCV_LIBRARIES} -L/opt/homebrew/opt/gcc/lib/gcc/current -lquadmath)
  endif()

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCV_${_uppercomponent} REQUIRED_VARS OPENCV_${_uppercomponent}_LIBRARY OPENCV_INCLUDE_DIR NAME_MISMATCHED)
  
  MARK_AS_ADVANCED(OPENCV_${_uppercomponent}_LIBRARY OPENCV_INCLUDE_DIR)
  
ENDIF(MSVC)
ENDMACRO(FIND_OPENCV_COMPONENT)

IF(COVISE_USE_OPENCV4)
    FIND_OPENCV_COMPONENT(aruco 470)
    FIND_OPENCV_COMPONENT(videoio 470)
    FIND_OPENCV_COMPONENT(core 470)
    FIND_OPENCV_COMPONENT(objdetect 470)
    FIND_OPENCV_COMPONENT(highgui 470)
    FIND_OPENCV_COMPONENT(imgproc 470)
    FIND_OPENCV_COMPONENT(imgcodecs 470)
    FIND_OPENCV_COMPONENT(calib3d 470)
    if(OPENCV_ARUCO_FOUND AND OPENCV_VIDEOIO_FOUND AND OPENCV_CORE_FOUND
            AND OPENCV_OBJDETECT_FOUND AND OPENCV_HIGHGUI_FOUND
            AND OPENCV_IMGPROC_FOUND AND OPENCV_CALIB3D_FOUND)
        set(OPENCV_FOUND TRUE)
    endif()
ELSEIF(COVISE_USE_OPENCV3)
    FIND_OPENCV_COMPONENT(aruco 310)
    FIND_OPENCV_COMPONENT(videoio 310)
    FIND_OPENCV_COMPONENT(core 310)
    FIND_OPENCV_COMPONENT(objdetect 310)
    FIND_OPENCV_COMPONENT(highgui 310)
    FIND_OPENCV_COMPONENT(imgproc 310)
    FIND_OPENCV_COMPONENT(imgcodecs 310)
    FIND_OPENCV_COMPONENT(calib3d 310)
    if(OPENCV_ARUCO_FOUND AND OPENCV_VIDEOIO_FOUND AND OPENCV_CORE_FOUND
            AND OPENCV_OBJDETECT_FOUND AND OPENCV_HIGHGUI_FOUND
            AND OPENCV_IMGPROC_FOUND AND OPENCV_CALIB3D_FOUND)
        set(OPENCV_FOUND TRUE)
    endif()
ELSE()
    FIND_OPENCV_COMPONENT(core 244)
    FIND_OPENCV_COMPONENT(objdetect 244)
    FIND_OPENCV_COMPONENT(highgui 244)
    FIND_OPENCV_COMPONENT(imgproc 244)
    if(OPENCV_CORE_FOUND AND OPENCV_OBJDETECT_FOUND AND OPENCV_HIGHGUI_FOUND AND OPENCV_IMGPROC_FOUND)
        set(OPENCV_FOUND TRUE)
    endif()
ENDIF()

IF(OPENCV_FOUND)
  SET(OPENCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIR})
ENDIF(OPENCV_FOUND)
