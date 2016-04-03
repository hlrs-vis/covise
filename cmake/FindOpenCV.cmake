# - Find OPENCV
# Find the OPENCV includes and library
#
#  OPENCV_INCLUDE_DIR - Where to find OPENCV includes
#  OPENCV_LIBRARIES   - List of libraries when using OPENCV
#  OPENCV_FOUND       - True if OPENCV was found

IF(OPENCV_INCLUDE_DIR)
  SET(OPENCV_FIND_QUIETLY TRUE)
ENDIF(OPENCV_INCLUDE_DIR)

IF(COVISE_USE_OPENCV3)
SET(OPENCV_EXTERNLIBS $ENV{EXTERNLIBS}/opencv3/)
ELSE(COVISE_USE_OPENCV3)
SET(OPENCV_EXTERNLIBS $ENV{EXTERNLIBS}/OpenCV/build/)
ENDIF(COVISE_USE_OPENCV3)


FIND_PATH(OPENCV_INCLUDE_DIR "opencv/cv.h"
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
  DOC "OpenCV - Headers"
  NO_DEFAULT_PATH
)
FIND_PATH(OPENCV_INCLUDE_DIR "opencv/cv.h"
  DOC "OpenCV - Headers"
)

MACRO(FIND_OPENCV_COMPONENT component version)

STRING(TOUPPER ${component} _uppercomponent)
SET(OPENCV_NAMES opencv_${component}${version} opencv_${component})
SET(OPENCV_WOV_NAMES opencv_${component})
SET(OPENCV_DBG_NAMES opencv_${component}${version}d)

FIND_LIBRARY(OPENCV_${_uppercomponent}_LIBRARY NAMES ${OPENCV_NAMES}
  PATHS
  ${OPENCV_EXTERNLIBS}
  $ENV{OPENCV_HOME}
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
  PATH_SUFFIXES lib lib64
  DOC "OPENCV - Library"
  NO_DEFAULT_PATH
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
    ${OPENCV_EXTERNLIBS}/x64/vc12/lib
    ${OPENCV_EXTERNLIBS}/x64/vc11/lib
    ${OPENCV_EXTERNLIBS}/x64/vc10/lib
    DOC "OPENCV - Library (Debug)"
  )
  
  IF(OPENCV_${_uppercomponent}_LIBRARY_DEBUG AND OPENCV_${_uppercomponent}_LIBRARY)
    SET(OPENCV_LIBRARIES ${OPENCV_LIBRARIES} optimized ${OPENCV_${_uppercomponent}_LIBRARY} debug ${OPENCV_${_uppercomponent}_LIBRARY_DEBUG})
  ENDIF(OPENCV_${_uppercomponent}_LIBRARY_DEBUG AND OPENCV_${_uppercomponent}_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCV DEFAULT_MSG OPENCV_${_uppercomponent}_LIBRARY OPENCV_${_uppercomponent}_LIBRARY_DEBUG OPENCV_INCLUDE_DIR)

  MARK_AS_ADVANCED(OPENCV_${_uppercomponent}_LIBRARY OPENCV_${_uppercomponent}_LIBRARY_DEBUG OPENCV_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world

  FIND_LIBRARY(OPENCV_${_uppercomponent}_LIBRARY NAMES ${OPENCV_WOV_NAMES}
    PATHS
    $ENV{OPENCV_HOME}/lib
    DOC "OPENCV - Library (WOV)"
  )

  SET(OPENCV_LIBRARIES ${OPENCV_LIBRARIES} ${OPENCV_${_uppercomponent}_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCV DEFAULT_MSG OPENCV_${_uppercomponent}_LIBRARY OPENCV_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OPENCV_${_uppercomponent}_LIBRARY OPENCV_INCLUDE_DIR)
  
ENDIF(MSVC)
ENDMACRO(FIND_OPENCV_COMPONENT)
IF(COVISE_USE_OPENCV3)
FIND_OPENCV_COMPONENT(aruco 310)
FIND_OPENCV_COMPONENT(videoio 310)
FIND_OPENCV_COMPONENT(core 310)
FIND_OPENCV_COMPONENT(objdetect 310)
FIND_OPENCV_COMPONENT(highgui 310)
FIND_OPENCV_COMPONENT(imgproc 310)
FIND_OPENCV_COMPONENT(calib3d 310)
ELSE(COVISE_USE_OPENCV3)
FIND_OPENCV_COMPONENT(core 244)
FIND_OPENCV_COMPONENT(objdetect 244)
FIND_OPENCV_COMPONENT(highgui 244)
FIND_OPENCV_COMPONENT(imgproc 244)
ENDIF(COVISE_USE_OPENCV3)

INCLUDE(FindPackageHandleStandardArgs)

IF(OPENCV_FOUND)
  SET(OPENCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIR})
ENDIF(OPENCV_FOUND)
