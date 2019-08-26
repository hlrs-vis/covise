# - Find LINPHONEDESKTOP
#
#  LINPHONEDESKTOP_INCLUDE_DIR - linphone include directory
#  LINPHONEDESKTOP_LIBRARIES_DIR - linphone libraries directory
#  LINPHONEDESKTOP_LIB_LIST - list of all libraries needed by linphone
#  LINPHONEDESKTOP_FOUND - linphone found

SET(LINPHONEDESKTOP_EXTERNLIBS $ENV{EXTERNLIBS}/linphone-desktop/)

# -----------------------------------------------------------------------------
# find paths

FIND_PATH(LINPHONEDESKTOP_INCLUDE_DIR linphone/core.h
  PATHS
  ${LINPHONEDESKTOP_EXTERNLIBS}/include
  /usr/local/include
  DOC "linphone-desktop - headers"
)

FIND_PATH(LINPHONEDESKTOP_LIBRARIES_DIR NAMES liblinphone.so
  PATHS
  ${LINPHONEDESKTOP_EXTERNLIBS}
  /usr/local
  PATH_SUFFIXES lib lib64
  DOC "linphone-desktop - libraries directory"
)

# -----------------------------------------------------------------------------
# find libraries

FOREACH(LIB_NAME liblinphone.so ortp pthread rt mediastreamer)

  FIND_LIBRARY(LINPHONEDESKTOP_${LIB_NAME}_LIB NAMES ${LIB_NAME}
    PATHS
    ${LINPHONEDESKTOP_EXTERNLIBS}
    /usr/local
    PATH_SUFFIXES lib lib64
    DOC "linphone-desktop - library used by linphone"
  )

  LIST(APPEND LIB_LIST ${LINPHONEDESKTOP_${LIB_NAME}_LIB})
  UNSET(${LINPHONEDESKTOP_${LIB_NAME}_LIB} CACHE)

ENDFOREACH()

SET(LINPHONEDESKTOP_LIB_LIST "${LIB_LIST}" CACHE STRING "linphone-desktop libraries")

INCLUDE(FindPackageHandleStandardArgs)

# -----------------------------------------------------------------------------
# finaly

IF(LINPHONEDESKTOP_LIBRARIES_DIR AND LINPHONEDESKTOP_INCLUDE_DIR)
  SET(LINPHONEDESKTOP_FOUND TRUE)

  message("-- Found Linphone LINPHONEDESKTOP_INCLUDE_DIR   = ${LINPHONEDESKTOP_INCLUDE_DIR}")
  message("                  LINPHONEDESKTOP_LIBRARIES_DIR = ${LINPHONEDESKTOP_LIBRARIES_DIR}")
  message("                  LINPHONEDESKTOP_LIB_LIST      = ${LINPHONEDESKTOP_LIB_LIST}")

ELSE()
  SET(LINPHONEDESKTOP_FOUND FALSE)
ENDIF()
