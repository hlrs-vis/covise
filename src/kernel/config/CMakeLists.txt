# @file
# 
# CMakeLists.txt for covise config
#
# @author Blasius Czink

set(CONFIGNEW_SOURCES config.cpp)
set(CONFIGNEW_HEADERS array.h access.h file.h value.h config.h)

set(PREFIX covconfig/)
include(${PREFIX}covconfig.cmake)

add_covise_library(coConfigNew ${CONFIGNEW_SOURCES} ${CONFIGNEW_HEADERS} ${COVCONFIG_SOURCES} ${COVCONFIG_HEADERS})
target_compile_definitions(coConfigNew PRIVATE CONFIG_NAMESPACE=covise COVCONFIG_COVISE)
target_include_directories(coConfigNew PRIVATE ${COVCONFIG_PRIVATE_INCLUDES})
target_link_libraries(coConfigNew PRIVATE ${COVCONFIG_PRIVATE_LIBRARIES})
covise_install_target(coConfigNew)
covise_install_headers(config ${COVCONFIG_HEADERS} ${CONFIGNEW_HEADERS})


using(Boost)

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
    coConfigString.h
    coConfigValue.h
)

COVISE_INSTALL_HEADERS(config ${CONFIG_HEADERS})
