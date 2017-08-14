USING(TrafficSimulation)


set(HEADERS
  SumoTraCI.h
  config.h
  windows_config.h
  foreign/tcpip/socket.h
  foreign/tcpip/storage.h
  traci-server/TraCIConstants.h
  traci-server/TraCIDefs.h
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

INCLUDE_DIRECTORIES(
)

cover_add_plugin(SumoTraCI ${HEADERS} ${SOURCES})
if(NOT MSVC)
    add_covise_compile_flags(SumoTraCI "-Wno-error=deprecated")
endif()
target_link_libraries(SumoTraCI ${EXTRA_LIBS})
