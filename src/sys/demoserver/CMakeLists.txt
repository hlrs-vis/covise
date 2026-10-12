SET(DEMOSERVER_SOURCES
  main.cpp
  demoserver.cpp
)

SET(DEMOSERVER_HEADER
  demoserver.h
)

ADD_COVISE_EXECUTABLE(demoserver ${DEMOSERVER_SOURCES} ${DEMOSERVER_HEADER})

target_link_libraries(demoserver 
                       coNet
                       nlohmann_json::nlohmann_json
                       Crow
                       hlrsDemoCommon
                       ) 

COVISE_INSTALL_TARGET(demoserver)

