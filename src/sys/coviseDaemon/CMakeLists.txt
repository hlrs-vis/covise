if (NOT COVISE_USE_QT)
    return()
endif()

USING(BOOST)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()
    
SET(REMOTE_LAUNCHER_SOURCES
  childOutput.cpp
  childProcess.cpp
  clientWidget.cpp
  coverDaemon.cpp
  coviseDaemon.cpp
  demoserver.cpp
  main.cpp
  mainWindow.cpp
  permissionRequest.cpp
  tui.cpp
  tuiCommands.cpp
)

SET(REMOTE_LAUNCHER_HEADER
  childOutput.h
  childProcess.h
  clientWidget.h
  coverDaemon.h
  coviseDaemon.h
  demoserver.h
  mainWindow.h
  metaTypes.h
  permissionRequest.h
  tui.h
  tuiCommands.h
)

SET(VRB_MOC_SRCS
  mainWindow.ui
  resources.qrc
)

ADD_COVISE_EXECUTABLE(coviseDaemon ${REMOTE_LAUNCHER_SOURCES} ${REMOTE_LAUNCHER_HEADER} ${VRB_MOC_SRCS})

target_link_libraries(coviseDaemon 
                       coMessages
                       coNet
                       coQtUtil
                       coUtil
                       coVRBClient
                       coConfigNew
                       nlohmann_json::nlohmann_json
                       Crow
                       hlrsDemoCommon
                       ) 
qt_use_modules(coviseDaemon Core Gui Widgets)

COVISE_INSTALL_TARGET(coviseDaemon)
