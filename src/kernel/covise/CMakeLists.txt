# @file
# 
# CMakeLists.txt for kernel - covise (coCore)
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_COVISE)
IF(WIN32)
  SET(EXTRA_LIBS iphlpapi)
ENDIF(WIN32)

SET(CORE_SOURCES
  covise_appproc.cpp
  covise_conv.cpp
  covise_converter.cpp
  covise_process.cpp
  covise_signal.cpp
  covise_time.cpp
  covise_version.cpp
  covise_msg.cpp
)

SET(CORE_HEADERS
  covise_time.h
  covise_appproc.h
  covise_objalg.h
  Covise_Util.h
  covise_version.h
  covise_converter.h
  covise_conv.h
  covise_process.h
  covise_signal.h
  covise_global.h
  covise_msg.h
  covise.h
)

# generate covise_version_autogen.h
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/covise_version_autogen.h.in" "${CMAKE_CURRENT_BINARY_DIR}/covise_version_autogen.h" @ONLY)

ADD_COVISE_LIBRARY(coCore ${COVISE_LIB_TYPE} ${CORE_SOURCES} ${CORE_HEADERS})
TARGET_LINK_LIBRARIES(coCore coConfig coNet coShm ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(coCore)
COVISE_INSTALL_HEADERS(covise ${CORE_HEADERS})
