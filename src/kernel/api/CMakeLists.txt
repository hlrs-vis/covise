# @file
# 
# CMakeLists.txt for kernel - api
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_API)

if (OPENSSL_FOUND)
INCLUDE_DIRECTORIES(
  "${OPENSSL_INCLUDE_DIR}"
)
add_definitions(-DHAVE_OPENSSL)
endif()

SET(API_SOURCES
  coBlankConv.cpp
  coBooleanParam.cpp
  coFileBrowserParam.cpp
  coColormapChoiceParam.cpp
  coMaterialParam.cpp
  coColormapParam.cpp
  coColorParam.cpp
  coChoiceParam.cpp
  coFeedback.cpp
  coCallFeedback.cpp
  coFloatParam.cpp
  coFloatSliderParam.cpp
  coFloatVectorParam.cpp
  coHideParam.cpp
  coInputPort.cpp
  coIntScalarParam.cpp
  coIntSliderParam.cpp
  coIntVectorParam.cpp
  coModule.cpp
  coOutputPort.cpp
  coPort.cpp
  coRenderer.cpp
  coSimpleModule.cpp
  coStepFile.cpp
  coStringParam.cpp
  coTimerParam.cpp
  coUifElem.cpp
  coUifPara.cpp
  coUifSwitchCase.cpp
  coUifSwitch.cpp
)

SET(API_HEADERS
  coBlankConv.h
  coBooleanParam.h
  coFileBrowserParam.h
  coColormapChoiceParam.h
  coMaterialParam.h
  coColormapParam.h
  coColorParam.h
  coChoiceParam.h
  coFeedback.h
  coCallFeedback.h
  coFloatParam.h
  coFloatSliderParam.h
  coFloatVectorParam.h
  coHideParam.h
  coInputPort.h
  coIntScalarParam.h
  coIntSliderParam.h
  coIntVectorParam.h
  coModule.h
  coOutputPort.h
  coPort.h
  coRenderer.h
  coSimpleModule.h
  coStepFile.h
  coStringParam.h
  coTimerParam.h
  coUifElem.h
  coUifPara.h
  coUifSwitchCase.h
  coUifSwitch.h
  
)

# simlib & simclient stuff
SET(SIMLIB_SOURCES
  coSimLib.cpp
  coSimClient.c
)

SET(SIMLIB_HEADERS
  coSimLib.h
  coSimLibComm.h
  coSimClient.h
)

IF(CMAKE_COMPILER_IS_GNUCC)
  SET_SOURCE_FILES_PROPERTIES(coSimClient.c PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

ADD_COVISE_LIBRARY(coApi ${COVISE_LIB_TYPE} ${API_SOURCES} ${API_HEADERS} ${SIMLIB_SOURCES} ${SIMLIB_HEADERS})
TARGET_LINK_LIBRARIES(coApi coAppl coUtil coCore coConfig)

COVISE_INSTALL_TARGET(coApi)
