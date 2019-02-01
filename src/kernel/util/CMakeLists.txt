# @file
# 
# CMakeLists.txt for kernel - util
#
# @author Blasius Czink

# MESSAGE("COVISEDIR = ${COVISEDIR}")
# MESSAGE("ARCHSUFFIX = ${ARCHSUFFIX}")
# MESSAGE("BASEARCHSUFFIX = ${BASEARCHSUFFIX}")
# COVISE_PRINT_LIB_SETUP(ZLIB)

# maybe change the following to use ${BASEARCHSUFFIX}
ADD_DEFINITIONS(-DUTIL_EXPORTS -DCOVISE_UTIL)

IF(WIN32)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DARCHSUFFIX=\\\"${BASEARCHSUFFIX}opt\\\"")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DARCHSUFFIX=\\\"${BASEARCHSUFFIX}opt\\\"")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DARCHSUFFIX=\\\"${BASEARCHSUFFIX}\\\"")
ELSE(WIN32)
  ADD_DEFINITIONS(-DARCHSUFFIX="${ARCHSUFFIX}")
ENDIF(WIN32)

IF(APPLE)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${APP_SERVICES_LIBRARY})
ENDIF(APPLE)

SET(UTIL_SOURCES
  covise_regexp.cpp
  covise_version.cpp
  coMatrix.cpp
  coSignal.cpp
  coStringTable.cpp
  coTimer.cpp
  Triangulator.cpp
  coLog.cpp
  string_util.cpp
  ChoiceList.cpp
  coVector.cpp
  UDP_Sender.cpp
  SerialCom.cpp
  coRestraint.cpp
  coErr.cpp
  coFileUtil.cpp
  coObjID.cpp
  coWristWatch.cpp
  environment.cpp
  coMaterial.cpp
)

SET(UTIL_HEADERS
  covise_regexp.h
  covise_version.h
  coSignal.h
  coStringTable.h
  coTimer.h
  coIdent.h
  coMatrix.h
  Triangulator.h
  coLog.h
  ChoiceList.h
  byteswap.h
  coDLList.h
  coErr.h
  coExport.h
  coFileUtil.h
  coHash.h
  coHashBase.h
  coHashIter.h
  coIntHash.h
  coIntMultiHash.h
  common.h
  coMultiHash.h
  coMultiHashBase.h
  coObjID.h
  coRestraint.h
  coTabletUIMessages.h
  coTypes.h
  coVector.h
  coviseCompat.h
  DLinkList.h
  unixcompat.h
  XGetopt.h
  UDP_Sender.h
  SerialCom.h
  coWristWatch.h
  covise_list.h
  environment.h
  coMaterial.h
)

SET(SYSDEP_HEADERS
  ../sysdep/khronos-glext.h
  ../sysdep/math.h
  ../sysdep/net.h
  ../sysdep/opengl.h
)

IF(MSVC)
  LIST(APPEND UTIL_SOURCES 
    regex/regerror.c
    regex/regfree.c
    regex/regcomp.c
    regex/regexec.c
  )
  SET(EXTRA_LIBS ${EXTRA_LIBS} wbemuuid)
ENDIF(MSVC)

IF(MINGW)
  LIST(APPEND UTIL_SOURCES 
    regex/regerror.c
    regex/regfree.c
    regex/regcomp.c
    regex/regexec.c
  )
#  SET(EXTRA_LIBS ${EXTRA_LIBS} regex)
ENDIF(MINGW)

# windows specific
IF(WIN32)
  LIST(APPEND UTIL_SOURCES XGetopt.cpp)
ENDIF(WIN32)

IF(WIN32)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ws2_32)
ENDIF(WIN32)

IF("${ARCHSUFFIX}" STREQUAL "mingw")
  SET(EXTRA_LIBS ${EXTRA_LIBS} stdc++)
ENDIF()

# generate covise_version_autogen.h
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/covise_version_autogen.h.in" "${CMAKE_CURRENT_BINARY_DIR}/covise_version_autogen.h" @ONLY)

ADD_COVISE_LIBRARY(coUtil ${COVISE_LIB_TYPE} ${UTIL_SOURCES} ${UTIL_HEADERS} ${SYSDEP_HEADERS})
TARGET_LINK_LIBRARIES(coUtil ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(coUtil)
COVISE_INSTALL_HEADERS(util ${UTIL_HEADERS})
COVISE_INSTALL_HEADERS(sysdep ${SYSDEP_HEADERS})
