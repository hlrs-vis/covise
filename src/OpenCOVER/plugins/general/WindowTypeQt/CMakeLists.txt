covise_find_package(X11)
if (X11_ICE_FOUND)
    include_directories(X11_ICE_INCLUDE_PATH)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

set(HEADERS
    QtOsgWidget.h
    QtView.h
    QtMainWindow.h
    WindowTypeQt.h
    KeyboardHelp.h
)

set(SOURCES
    QtOsgWidget.cpp
    QtView.cpp
    QtMainWindow.cpp
    WindowTypeQt.cpp
    KeyboardHelp.cpp
    KeyboardHelp.ui
)

cover_add_plugin(WindowTypeQt ${HEADERS} ${SOURCES})
qt5_use_modules(WindowTypeQt Core Widgets OpenGL)

if (X11_ICE_FOUND)
    target_link_libraries(WindowTypeQt ${X11_ICE_LIB})
endif()
