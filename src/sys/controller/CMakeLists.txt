# @file
# 
# CMakeLists.txt for sys controller
#
# @author Blasius Czink

IF(COVISE_BUILD_WEBSERVICE)
  ADD_DEFINITIONS(-DHAVE_GSOAP)
ENDIF()

SET(CONTROLLER_SOURCES
    Token.cpp
	CTRLMain.cpp
	CTRLHandler.cpp
	CTRLGlobal.cpp
	covise_module.cpp
	control_process.cpp
	control_coviseconfig.cpp
	control_modlist.cpp
	control_module.cpp
	control_netmod.cpp
	control_object.cpp
	control_port.cpp
	AccessGridDaemon.cpp
)

SET(CONTROLLER_HEADERS
    Token.h
	CTRLHandler.h
	CTRLGlobal.h
	covise_module.h
	control_process.h
	control_coviseconfig.h
	control_modlist.h
	control_module.h
	control_netmod.h
	control_object.h
	control_port.h
	AccessGridDaemon.h
)

if(OPENSSL_FOUND)
   INCLUDE_DIRECTORIES(
      ../CoviseSSLDaemon/CoviseClient
   )
   ADD_DEFINITIONS(-DHAVE_OPENSSL)

   set(CONTROLLER_HEADERS ${CONTROLLER_SOURCES} 
      ../CoviseSSLDaemon/CoviseClient/SSLClient.cpp
   )
   set(CONTROLLER_HEADERS ${CONTROLLER_HEADERS} 
      ../CoviseSSLDaemon/CoviseClient/SSLClient.h
   )
endif()

ADD_COVISE_EXECUTABLE(covise ${CONTROLLER_SOURCES} ${CONTROLLER_HEADERS})
TARGET_LINK_LIBRARIES(covise coAppl coApi coCore coConfig coUtil coVRBClient) 
qt_use_modules(covise Core)

COVISE_INSTALL_TARGET(covise)
