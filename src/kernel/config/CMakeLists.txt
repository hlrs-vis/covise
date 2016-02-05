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
    coConfigBool.h
    coConfigConstants.h
    coConfigEditorController.h
    coConfigEntry.h
    coConfigEntryPtrList.h
    coConfigEntryString.h
    coConfigEntryToEditor.h
    coConfigFloat.h
    coConfigGroup.h
    coConfig.h
    coConfigImportReader.h
    coConfigInt.h
    coConfigLog.h
    coConfigLong.h
    coConfigRoot.h
    coConfigSchemaInfos.h
    coConfigSchemaInfosList.h
    coConfigString.h
    coConfigValue.h
    CoviseConfig.h
)

COVISE_INSTALL_HEADERS(config ${CONFIG_HEADERS})
