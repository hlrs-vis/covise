# @file
# 
# CMakeLists.txt for covise config
#
# @author Blasius Czink

ADD_SUBDIRECTORY(kernel)
IF(COVISE_BUILD)
   ADD_SUBDIRECTORY(coEditor)
ENDIF(COVISE_BUILD)

set(CONFIG_HEADERS
    CoviseConfig.h
    coConfig.h
    coConfigBool.h
    coConfigConstants.h
    coConfigEditorController.h
    coConfigEntry.h
    coConfigEntryString.h
    coConfigEntryToEditor.h
    coConfigFile.h
    coConfigFloat.h
    coConfigGroup.h
    coConfigInt.h
    coConfigLog.h
    coConfigLong.h
    coConfigRoot.h
    coConfigSchemaInfos.h
    coConfigSchemaInfosList.h
    coConfigString.h
    coConfigValue.h
)

COVISE_INSTALL_HEADERS(config ${CONFIG_HEADERS})
