if (NOT COVISE_USE_QT)
    return()
endif()

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()
    

set(SOURCES
    demowindow.cpp
    flowlayout.cpp
    launch.cpp
    main.cpp
    tui.cpp
    verticallabel.cpp
)

set(HEADERS
    demowindow.h
    flowlayout.h
    launch.h
    tui.h
    verticallabel.h
)



ADD_COVISE_EXECUTABLE(hlrsDemo)
target_link_libraries(hlrsDemo
    nlohmann_json::nlohmann_json
    hlrsDemoCommon
)
qt_use_modules(hlrsDemo Core Gui Widgets)

COVISE_INSTALL_TARGET(hlrsDemo)
