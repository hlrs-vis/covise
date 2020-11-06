
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
    
SET(REMOTE_LAUNCHER_SOURCES
  main.cpp
  mainWindow.cpp
  tui.cpp
  tuiCommands.cpp
)

SET(REMOTE_LAUNCHER_HEADER
  mainWindow.h
  tui.h
  tuiCommands.h
)

SET(VRB_MOC_SRCS
  mainWindow.ui
  resources.qrc
)

ADD_COVISE_EXECUTABLE(VrbRemoteLauncher ${REMOTE_LAUNCHER_SOURCES} ${REMOTE_LAUNCHER_HEADER} ${VRB_MOC_SRCS})

target_link_libraries(VrbRemoteLauncher 
                       Qt5::Core
                       Qt5::Gui
                       Qt5::Widgets
                       coVRBRemoteLauncher
                       ) 

COVISE_INSTALL_TARGET(VrbRemoteLauncher)
