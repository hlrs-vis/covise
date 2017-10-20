set(CMAKE_AUTOMOC ON)

set(HEADERS
    QtOsgWidget.h
    QtView.h
    WindowTypeQt.h
)

set(SOURCES
    QtOsgWidget.cpp
    QtView.cpp
    WindowTypeQt.cpp
)

cover_add_plugin(WindowTypeQt ${HEADERS} ${SOURCES})
qt_use_modules(WindowTypeQt Core Widgets OpenGL)
