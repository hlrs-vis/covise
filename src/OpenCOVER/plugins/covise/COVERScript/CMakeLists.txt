#OpenCOVER plugins general COVERScript

USING(GSOAP)

IF(NOT COVISE_BUILD_WEBSERVICE)
  RETURN()
ENDIF()

ADD_DEFINITIONS(-DWITH_NONAMESPACES)


SET(HEADERS
  CoviseEventMessageSerialiser.h
  ScriptDebugger.h
  ScriptEngineProvider.h
  ScriptInterface.h
  ScriptPlugin.h
  ScriptTabletUI.h
  ScriptVrmlNode.h
  ScriptWsCovise.h
)

SET(MOC_HEADERS
  DynamicUI.h
  ScriptDebugger.h
  ScriptInterface.h
  ScriptTabletUI.h
  ScriptVrmlNode.h
  ScriptWsCovise.h
)

SET(SOURCES
  CoviseEventMessageSerialiser.cpp
  DynamicUI.cpp
  ScriptDebugger.cpp
  ScriptEngineProvider.cpp
  ScriptInterface.cpp
  ScriptPlugin.cpp
  ScriptTabletUI.cpp
  ScriptVrmlNode.cpp
  ScriptWsCovise.cpp
)

QT_WRAP_CPP(MOC_SRCS ${MOC_HEADERS})

LIST(APPEND SOURCES ${MOC_SRCS})
LIST(APPEND HEADERS ${MOC_HEADERS})

cover_add_plugin(COVERScript Vrml97 ${COVISE_WEBSERVICE_LIBRARY} ${COVISE_APPL_LIBRARY} ${COVISE_CORE_LIBRARY} ${COVISE_SHM_LIBRARY})
qt_use_modules(COVERScript Script ScriptTools Xml Widgets)
