# @file
# 
# CMakeLists.txt for sys ScriptingInterface (_covise)
#
# @author Blasius Czink

SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "$ENV{EXTERNLIBS}/swig")
COVISE_FIND_PACKAGE(SWIG)
IF(NOT SWIG_FOUND)
  MESSAGE("SWIG Executable not found! Skipping ScriptingInterface (_covise)")
  RETURN()
ENDIF()
INCLUDE(${SWIG_USE_FILE})

IF(NOT PYTHON_FOUND)
  MESSAGE("Python not found! Skipping ScriptingInterface (_covise)")
  RETURN()
ENDIF()

IF(WIN32)
  REMOVE_DEFINITIONS(-DDEBUG)
ENDIF(WIN32)

IF(MSVC)
  # arrrgh pyconfig.h contains a #pragma comment(lib,"python27_d.lib")
  GET_FILENAME_COMPONENT(PYTHON_LIB_DIR "${PYTHON_DEBUG_LIBRARY}" PATH)
  LINK_DIRECTORIES(${PYTHON_LIB_DIR})
ENDIF(MSVC)

SET(CMAKE_SWIG_FLAGS "")
SET(SWIGSOURCE "${CMAKE_CURRENT_SOURCE_DIR}/scriptInterface.i")
SET(SCRIPTING_SRC "${CMAKE_CURRENT_SOURCE_DIR}/scriptInterface.cpp")

SET_SOURCE_FILES_PROPERTIES(${SWIGSOURCE} PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(${SWIGSOURCE} PROPERTIES SWIG_FLAGS "-modern;-py3")

if (${CMAKE_VERSION} VERSION_LESS "3.13")
  SWIG_ADD_MODULE(covise python ${SWIGSOURCE} ${SCRIPTING_SRC})
else()
  SWIG_ADD_LIBRARY(covise LANGUAGE python TYPE MODULE SOURCES ${SWIGSOURCE} ${SCRIPTING_SRC})
endif()
# ${SWIG_MODULE_covise_REAL_NAME} contains actual targetname
COVISE_ADJUST_OUTPUT_DIR(${SWIG_MODULE_covise_REAL_NAME})
SWIG_LINK_LIBRARIES(covise coAppl Python::Python)

# for VisualStudio it seems we have to rename debug versions to *_d.pyd
IF(MSVC)
    IF("${COVISE_ARCHSUFFIX}" STREQUAL "${BASEARCHSUFFIX}")
    SET_TARGET_PROPERTIES(${SWIG_MODULE_covise_REAL_NAME} PROPERTIES OUTPUT_NAME_DEBUG "_covise_d")
  ENDIF()
ENDIF(MSVC)

# place covise.py in the correct directory
IF(MSVC)
  IF(upper_conf_type_str STREQUAL "DEBUG")
  ADD_CUSTOM_COMMAND(TARGET ${SWIG_MODULE_covise_REAL_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/covise.py ${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}/lib/covise.py
  )
  ELSE()
  ADD_CUSTOM_COMMAND(TARGET ${SWIG_MODULE_covise_REAL_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/covise.py ${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}opt/lib/covise.py
  )
  ENDIF()
ELSE(MSVC)
  ADD_CUSTOM_COMMAND(TARGET ${SWIG_MODULE_covise_REAL_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/covise.py ${COVISEDIR}/${COVISE_ARCHSUFFIX}/lib/covise.py
  )
ENDIF(MSVC)
# place scriptingInterface.py in the output directory
ADD_CUSTOM_COMMAND(TARGET ${SWIG_MODULE_covise_REAL_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/covise.py scriptInterface.py
)
