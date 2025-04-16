# @file
# 
# CMakeLists.txt for kernel - util
#
# @author Blasius Czink

# MESSAGE("COVISEDIR = ${COVISEDIR}")
# MESSAGE("ARCHSUFFIX = ${COVISE_ARCHSUFFIX}")
# MESSAGE("BASEARCHSUFFIX = ${BASEARCHSUFFIX}")
# COVISE_PRINT_LIB_SETUP(ZLIB)

# maybe change the following to use ${BASEARCHSUFFIX}
ADD_DEFINITIONS(-DUTIL_EXPORTS -DCOVISE_UTIL)

USING(Pthreads)

ADD_DEFINITIONS(-DARCHSUFFIX="${BASEARCHSUFFIX}$<IF:$<CONFIG:Debug>,,opt>")

IF(APPLE)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${APP_SERVICES_LIBRARY})
ENDIF(APPLE)

SET(UTIL_SOURCES
  ChoiceList.cpp
  SerialCom.cpp
  Triangulator.cpp
  UDPComm.cpp
  UDP_Sender.cpp
  asyncWait.cpp
  coErr.cpp
  coException.cpp
  coFileUtil.cpp
  coLog.cpp
  coMaterial.cpp
  coMatrix.cpp
  coObjID.cpp
  coRestraint.cpp
  coSignal.cpp
  coSpawnProgram.cpp
  coStringTable.cpp
  coTimer.cpp
  coVector.cpp
  coWristWatch.cpp
  covise_regexp.cpp
  covise_version.cpp
  environment.cpp
  string_util.cpp
  threadname.cpp
)

SET(UTIL_HEADERS
  ChoiceList.h
  DLinkList.h
  SerialCom.h
  Triangulator.h
  UDPComm.h
  UDP_Sender.h
  XGetopt.h
  asyncWait.h
  byteswap.h
  coErr.h
  coException.h
  coExport.h
  coFileUtil.h
  coHash.h
  coHashBase.h
  coHashIter.h
  coIntHash.h
  coIntMultiHash.h
  coLog.h
  coMaterial.h
  coMatrix.h
  coMultiHash.h
  coMultiHashBase.h
  coObjID.h
  coRestraint.h
  coSignal.h
  coSpawnProgram.h
  coStringTable.h
  coTabletUIMessages.h
  coTimer.h
  coTypes.h
  coVector.h
  coWristWatch.h
  common.h
  coviseCompat.h
  covise_list.h
  covise_regexp.h
  covise_version.h
  environment.h
  threadname.h
  tryPrint.h
  unixcompat.h
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

IF("${COVISE_ARCHSUFFIX}" STREQUAL "mingw")
  SET(EXTRA_LIBS ${EXTRA_LIBS} stdc++)
ENDIF()

# generate covise_version_autogen.h
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/covise_version_autogen.h.in" "${CMAKE_CURRENT_BINARY_DIR}/covise_version_autogen.h" @ONLY)

ADD_COVISE_LIBRARY(coUtil ${COVISE_LIB_TYPE} ${UTIL_SOURCES} ${UTIL_HEADERS} ${SYSDEP_HEADERS})
TARGET_LINK_LIBRARIES(coUtil ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(coUtil)
COVISE_INSTALL_HEADERS(util ${UTIL_HEADERS})
COVISE_INSTALL_HEADERS(sysdep ${SYSDEP_HEADERS})
