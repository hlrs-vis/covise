covise_find_package(X11)
if (X11_ICE_FOUND)
    include_directories(X11_ICE_INCLUDE_PATH)
endif()

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

if (X11_ICE_FOUND)
    target_link_libraries(WindowTypeQt ${X11_ICE_LIB})
endif()
