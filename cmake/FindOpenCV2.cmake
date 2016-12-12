# - Find OPENCV2
# Find the OPENCV2 includes and library
#
#  OPENCV2_INCLUDE_DIR - Where to find OPENCV2 includes
#  OPENCV2_LIBRARIES   - List of libraries when using OPENCV2
#  OPENCV2_FOUND       - True if OPENCV2 was found

IF(OPENCV2_INCLUDE_DIR)
  SET(OPENCV2_FIND_QUIETLY TRUE)
ENDIF(OPENCV2_INCLUDE_DIR)

SET(OPENCV2_EXTERNLIBS $ENV{EXTERNLIBS}/opencv2/)

FIND_PATH(OPENCV2_INCLUDE_DIR "opencv/cv.h"
  PATHS
  ${OPENCV2_EXTERNLIBS}/include
  $ENV{OPENCV2_HOME}/include
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
FIND_PATH(OPENCV2_INCLUDE_DIR "opencv/cv.h"
  DOC "OpenCV - Headers"
)

INCLUDE(FindPackageHandleStandardArgs)

MACRO(FIND_OPENCV2_COMPONENT component version)

STRING(TOUPPER ${component} _uppercomponent)
SET(OPENCV2_NAMES opencv_${component}${version} opencv_${component})
SET(OPENCV2_WOV_NAMES opencv_${component})
SET(OPENCV2_DBG_NAMES opencv_${component}${version}d)

FIND_LIBRARY(OPENCV2_${_uppercomponent}_LIBRARY NAMES ${OPENCV2_NAMES}
  PATHS
  ${OPENCV2_EXTERNLIBS}
  $ENV{OPENCV2_HOME}
  ${OPENCV2_EXTERNLIBS}/x64/vc14
  ${OPENCV2_EXTERNLIBS}/x64/vc12
  ${OPENCV2_EXTERNLIBS}/x64/vc11
  ${OPENCV2_EXTERNLIBS}/x64/vc10
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  PATH_SUFFIXES lib lib64
  DOC "OPENCV2 - Library"
  NO_DEFAULT_PATH
)
FIND_LIBRARY(OPENCV2_${_uppercomponent}_LIBRARY NAMES ${OPENCV2_NAMES}
  DOC "OPENCV2 - Library"
)

IF(MSVC)
  # VisualStudio needs a debug version
  #MESSAGE(${OPENCV2_DBG_NAMES})
  FIND_LIBRARY(OPENCV2_${_uppercomponent}_LIBRARY_DEBUG NAMES ${OPENCV2_DBG_NAMES}
    PATHS
    $ENV{OPENCV2_HOME}/lib
    ${OPENCV2_EXTERNLIBS}/x64/vc14/lib
    ${OPENCV2_EXTERNLIBS}/x64/vc12/lib
    ${OPENCV2_EXTERNLIBS}/x64/vc11/lib
    ${OPENCV2_EXTERNLIBS}/x64/vc10/lib
    DOC "OPENCV2 - Library (Debug)"
  )
  
  IF(OPENCV2_${_uppercomponent}_LIBRARY_DEBUG AND OPENCV2_${_uppercomponent}_LIBRARY)
    SET(OPENCV2_LIBRARIES ${OPENCV2_LIBRARIES} optimized ${OPENCV2_${_uppercomponent}_LIBRARY} debug ${OPENCV2_${_uppercomponent}_LIBRARY_DEBUG})
  ENDIF(OPENCV2_${_uppercomponent}_LIBRARY_DEBUG AND OPENCV2_${_uppercomponent}_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCV2_${_uppercomponent} DEFAULT_MSG OPENCV2_${_uppercomponent}_LIBRARY OPENCV2_${_uppercomponent}_LIBRARY_DEBUG OPENCV2_INCLUDE_DIR)

  MARK_AS_ADVANCED(OPENCV2_${_uppercomponent}_LIBRARY OPENCV2_${_uppercomponent}_LIBRARY_DEBUG OPENCV2_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world

  FIND_LIBRARY(OPENCV2_${_uppercomponent}_LIBRARY NAMES ${OPENCV2_WOV_NAMES}
    PATHS
    $ENV{OPENCV2_HOME}/lib
    DOC "OPENCV2 - Library (WOV)"
  )

  SET(OPENCV2_LIBRARIES ${OPENCV2_LIBRARIES} ${OPENCV2_${_uppercomponent}_LIBRARY})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCV2_${_uppercomponent} DEFAULT_MSG OPENCV2_${_uppercomponent}_LIBRARY OPENCV2_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(OPENCV2_${_uppercomponent}_LIBRARY OPENCV2_INCLUDE_DIR)
  
ENDIF(MSVC)
ENDMACRO(FIND_OPENCV2_COMPONENT)

FIND_OPENCV2_COMPONENT(core 2412)
FIND_OPENCV2_COMPONENT(objdetect 2412)
FIND_OPENCV2_COMPONENT(highgui 2412)
FIND_OPENCV2_COMPONENT(imgproc 2412)
if(OPENCV2_CORE_FOUND AND OPENCV2_OBJDETECT_FOUND AND OPENCV2_HIGHGUI_FOUND AND OPENCV2_IMGPROC_FOUND)
  set(OPENCV2_FOUND TRUE)
endif()

IF(OPENCV2_FOUND)
  SET(OPENCV2_INCLUDE_DIRS ${OPENCV2_INCLUDE_DIR})
ENDIF(OPENCV2_FOUND)
