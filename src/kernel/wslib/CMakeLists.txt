# @file
# 
# CMakeLists.txt for kernel - wslib (coWS)
#
# @author Blasius Czink

IF(NOT COVISE_BUILD_WEBSERVICE)
  RETURN()
ENDIF()

USING(GSOAP)

ADD_DEFINITIONS(-DWSLIB_EXPORT -DWITH_NONAMESPACES)
# ADD_DEFINITIONS(-DSOAP_DEBUG -DDEBUG)

SET(GSOAP_STUBS_DIR "${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/WS/wsdl.qrc.in" "${CMAKE_CURRENT_BINARY_DIR}/wsdl.qrc" @ONLY)
QT_ADD_RESOURCES(WSLIB_RESOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/wsdl.qrc" OPTIONS "-no-compress")

INCLUDE_DIRECTORIES(
  ${GSOAP_INCLUDE_DIR}
)

SET(WSLIB_SOURCES
  WSCOVISE.cpp
  WSCOVISEService.cpp
  WSCOVISEClient.cpp
  WSStdSOAP2.cpp
  WSMap.cpp
  WSModule.cpp
  WSPort.cpp
  WSParameter.cpp
  WSBooleanParameter.cpp
  WSChoiceParameter.cpp
  WSColormapChoiceParameter.cpp
  WSFileBrowserParameter.cpp
  WSFloatScalarParameter.cpp
  WSFloatSliderParameter.cpp
  WSFloatVectorParameter.cpp
  WSIntScalarParameter.cpp
  WSIntSliderParameter.cpp
  WSIntVectorParameter.cpp
  WSStringParameter.cpp
  WSLink.cpp
  WSTools.cpp
  WSCOVISEServiceWrapper.cpp
  WSCOVISEProxyWrapper.cpp
  WSCoviseCWrapper.cpp
  WSColormap.cpp
)

SET(WSLIB_HEADERS
  WSTools.h
  WSCoviseStub.h
  WSColormap.h
  WSColorParameter.h
  WS/wscovise.h
  WS/wscoviseevents.h
)

SET(WSLIB_MOC_HEADERS
  WSCOVISEClient.h
  WSMap.h
  WSModule.h
  WSPort.h
  WSParameter.h
  WSBooleanParameter.h
  WSChoiceParameter.h
  WSColormapChoiceParameter.h
  WSFileBrowserParameter.h
  WSFloatScalarParameter.h
  WSFloatSliderParameter.h
  WSFloatVectorParameter.h
  WSIntScalarParameter.h
  WSIntSliderParameter.h
  WSIntVectorParameter.h
  WSStringParameter.h
  WSLink.h
)

QT_WRAP_CPP(WSLIB_MOC_SRCS ${WSLIB_MOC_HEADERS})

# gsoap empty env.h
GSOAP_TARGET("WS/env.h" WSLIB_ENV_SOAP)

# gsoap covise service
GSOAP_TARGET("WS/wscovise.h" WSLIB_SOAP)

# since the actual stub is wrapped make sure cmake gets the dependencies right
SET_SOURCE_FILES_PROPERTIES(WSCoviseStub.h PROPERTIES OBJECT_DEPENDS ${WSLIB_SOAP_STUBS})

# make sure Qt's resource compiler is run after gsoap generated the given wsdl and xsd files
SET_SOURCE_FILES_PROPERTIES(${WSLIB_RESOURCE_FILES} PROPERTIES OBJECT_DEPENDS ${WSLIB_SOAP_STUBS})

# message("WSLIB_SOAP_STUBS = ${WSLIB_SOAP_STUBS}")
# message("WSLIB_SOAP_CLIENT = ${WSLIB_SOAP_CLIENT}")
# message("WSLIB_SOAP_SERVER = ${WSLIB_SOAP_SERVER}")
# message("WSLIB_SOAP_HEADER = ${WSLIB_SOAP_HEADER}")

ADD_COVISE_LIBRARY(coWS ${COVISE_LIB_TYPE} ${WSLIB_SOURCES} ${WSLIB_HEADERS} ${WSLIB_MOC_HEADERS} ${WSLIB_MOC_SRCS} ${WSLIB_RESOURCE_FILES} ${WSLIB_ENV_SOAP_STUBS})
ADD_CUSTOM_COMMAND(TARGET coWS
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${GSOAP_STUBS_DIR}/coviseStub.h ${COVISE_DESTDIR}/${COVISE_ARCHSUFFIX}/include/wslib/coviseStub.h)
qt_use_modules(coWS Xml Network)
COVISE_WNOERROR(coWS)
TARGET_LINK_LIBRARIES(coWS coDo coCore coUtil coConfig) # ${CMAKE_THREAD_LIBS_INIT}

COVISE_INSTALL_TARGET(coWS)
COVISE_INSTALL_HEADERS(ws ${WSLIB_HEADERS})
