#cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
#find_package(COVER REQUIRED)

set(HEADERS
  SumoTraCI.h
  config.h
  windows_config.h
  foreign/tcpip/socket.h
  foreign/tcpip/storage.h
  traci-server/TraCIConstants.h
  utils/common/SUMOTime.h
  utils/common/StdDefs.h
  utils/common/UtilExceptions.h
  utils/common/TplConvert.h
  utils/traci/TraCIAPI.h

)
set(SOURCES
  SumoTraCI.cpp
  foreign/tcpip/socket.cpp
  foreign/tcpip/storage.cpp
  utils/common/SUMOTime.cpp
  utils/traci/TraCIAPI.cpp
)

cover_add_plugin(SumoTraCI ${HEADERS} ${SOURCES})
