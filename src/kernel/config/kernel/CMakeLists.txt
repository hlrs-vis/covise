# @file
# 
# CMakeLists.txt for config-kernel
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCONFIG_EXPORT)
using(boost)
INCLUDE_DIRECTORIES(
  "${XERCESC_INCLUDE_DIR}"
  ..
)

SET(CONFIG_SOURCES
  CoviseConfig.cpp
  coConfig.cpp
  coConfigBool.cpp
  coConfigConstants.cpp
  coConfigEntry.cpp
  coConfigEntryString.cpp
  coConfigEntryToEditor.cpp
  coConfigFile.cpp
  coConfigFloat.cpp
  coConfigGroup.cpp
  coConfigInt.cpp
  coConfigLong.cpp
  coConfigRoot.cpp
  coConfigRootErrorHandler.cpp
  coConfigSchema.cpp
  coConfigSchemaInfos.cpp
  coConfigString.cpp
  coConfigTools.cpp
  coConfigXercesConverter.cpp
)

SET(CONFIG_HEADERS
  coConfigRootErrorHandler.h
  coConfigSchema.h
  coConfigTools.h
  coConfigXercesConverter.h
  coConfigXercesRoot.h
)

SET(CONFIG_DEV_HEADERS
  ../CoviseConfig.h
  ../coConfig.h
  ../coConfigBool.h
  ../coConfigConstants.h
  ../coConfigEditorController.h
  ../coConfigEntry.h
  ../coConfigEntryString.h
  ../coConfigEntryToEditor.h
  ../coConfigFloat.h
  ../coConfigGroup.h
  ../coConfigInt.h
  ../coConfigLog.h
  ../coConfigLong.h
  ../coConfigRoot.h
  ../coConfigSchemaInfos.h
  ../coConfigString.h
  ../coConfigValue.h
)

ADD_COVISE_LIBRARY(coConfig ${COVISE_LIB_TYPE} ${CONFIG_SOURCES} ${CONFIG_HEADERS} ${CONFIG_DEV_HEADERS})
# maybe move the following to CovisePresets.cmake
IF(MSVC80 OR MSVC90)
   ADD_COVISE_COMPILE_FLAGS(coConfig "/Zc:wchar_t-")
ENDIF()
IF(WIN32)
   SET(EXTRA_LIBS ${EXTRA_LIBS} ws2_32)
ENDIF(WIN32)
IF(MSVC)
   SET(EXTRA_LIBS ${EXTRA_LIBS} wbemuuid)
ENDIF(MSVC)
TARGET_LINK_LIBRARIES(coConfig coUtil ${EXTRA_LIBS} ${XERCESC_LIBRARIES})
COVISE_WNOERROR(coConfig BASEARCH goddard)
COVISE_INSTALL_TARGET(coConfig)


