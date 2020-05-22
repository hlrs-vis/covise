# @file
# 
# CMakeLists.txt for config-kernel
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCONFIG_EXPORT)

INCLUDE_DIRECTORIES(
  "${XERCESC_INCLUDE_DIR}"
  ..
)

SET(CONFIG_SOURCES
  coConfigLog.cpp
  coConfigEntryToEditor.cpp
  coConfigEntryPtrList.cpp
  coConfigEntryString.cpp
  coConfigEntry.cpp
  coConfigRoot.cpp
  coConfigRootErrorHandler.cpp
  coConfigGroup.cpp
  coConfigConstants.cpp
  coConfig.cpp
  coConfigBool.cpp
  coConfigString.cpp
  coConfigFloat.cpp
  coConfigInt.cpp
  coConfigImportReader.cpp
  coConfigLong.cpp
  CoviseConfig.cpp
  coConfigSchema.cpp
  coConfigSchemaInfos.cpp
  coConfigTools.cpp
)

SET(CONFIG_HEADERS
  coConfigRootErrorHandler.h
  coConfigXercesRoot.h
  coConfigSchema.h
  coConfigTools.h
)

SET(CONFIG_DEV_HEADERS
  ../coConfig.h
  ../coConfigBool.h
  ../coConfigConstants.h
  ../coConfigEditorController.h
  ../coConfigEntry.h
  ../coConfigEntryPtrList.h
  ../coConfigEntryString.h
  ../coConfigEntryToEditor.h
  ../coConfigFloat.h
  ../coConfigGroup.h
  ../coConfigImportReader.h
  ../coConfigInt.h
  ../coConfigLog.h
  ../coConfigLong.h
  ../coConfigRoot.h
  ../coConfigSchemaInfos.h
  ../coConfigSchemaInfosList.h
  ../coConfigString.h
  ../coConfigValue.h
  ../CoviseConfig.h
)

ADD_COVISE_LIBRARY(coConfig ${COVISE_LIB_TYPE} ${CONFIG_SOURCES} ${CONFIG_HEADERS} ${CONFIG_DEV_HEADERS})
qt_use_modules(coConfig Core Xml)
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
TARGET_LINK_LIBRARIES(coConfig ${EXTRA_LIBS} ${XERCESC_LIBRARIES})
COVISE_WNOERROR(coConfig BASEARCH goddard)
COVISE_INSTALL_TARGET(coConfig)
