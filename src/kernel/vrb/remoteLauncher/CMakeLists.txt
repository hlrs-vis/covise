#coVRBRemoteLauncher
COVISE_FIND_PACKAGE(Qt5 REQUIRED 
                COMPONENTS 
                    Core
                    Gui
                    Widgets
)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

SET(CO_REMOTE_LAUNCHER_SOURCES
  ClientWidget.cpp
  VrbRemoteLauncher.cpp
  SpawnProgram.cpp
)

SET(CO_REMOTE_LAUNCHER_HEADER
  ClientWidget.h
  VrbRemoteLauncher.h
  MessageTypes.h
  SpawnProgram.h
  export.h
)

ADD_COVISE_LIBRARY(coVRBRemoteLauncher ${CO_REMOTE_LAUNCHER_SOURCES} ${CO_REMOTE_LAUNCHER_HEADER})
target_link_libraries(coVRBRemoteLauncher 
                       Qt5::Gui
                       Qt5::Widgets
                       Qt5::Core
                       coConfig
                       coNet
                       coUtil
                       coVRBClient
                       ) 

COVISE_INSTALL_TARGET(coVRBRemoteLauncher)     







