
COVISE_FIND_PACKAGE(Qt5 REQUIRED 
                COMPONENTS 
                    Core
                    Gui
                    Widgets
)

USING(BOOST)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()
    
SET(REMOTE_LAUNCHER_SOURCES
  clientWidget.cpp
  coverDaemon.cpp
  coviseDaemon.cpp
  main.cpp
  mainWindow.cpp
  tui.cpp
  tuiCommands.cpp
)

SET(REMOTE_LAUNCHER_HEADER
  clientWidget.h
  coverDaemon.h
  coviseDaemon.h
  mainWindow.h
  metaTypes.h
  tui.h
  tuiCommands.h
)

SET(VRB_MOC_SRCS
  mainWindow.ui
  resources.qrc
)

ADD_COVISE_EXECUTABLE(coviseDaemon ${REMOTE_LAUNCHER_SOURCES} ${REMOTE_LAUNCHER_HEADER} ${VRB_MOC_SRCS})

target_link_libraries(coviseDaemon 
                       Qt5::Core
                       Qt5::Gui
                       Qt5::Widgets
                       coUtil
                       coVRBClient
                       coMessages
                       ) 

COVISE_INSTALL_TARGET(coviseDaemon)
