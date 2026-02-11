if(NOT COVISE_USE_QT)
    return()
endif()
if (COVISE_USE_QT5 AND Qt5OpenGL_VERSION VERSION_LESS 5.4.0)
    return()
endif()

if (COVISE_USE_X11)
    covise_find_package(X11)
    if (X11_FOUND)
        include_directories(X11_X11_INCLUDE_PATH)
        if (X11_ICE_FOUND)
            include_directories(${X11_ICE_INCLUDE_PATH})
            add_definitions(-DUSE_X11_ICE)
        endif()
    endif()
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
    QtOsgWidgetImpl.cpp
    QtView.cpp
    QtMainWindow.cpp
    WindowTypeQt.cpp
    KeyboardHelp.cpp
    KeyboardHelp.ui
    AboutDialog.ui
)

qt_add_resources(QRC_SOURCES about.qrc icons.qrc)
qt_wrap_ui(UI_SOURCES AboutDialog.ui KeyboardHelp.ui)
cover_add_plugin(WindowTypeQt ${HEADERS} ${SOURCES} ${QRC_SOURCES} ${UI_SOURCES})
qt_use_modules(WindowTypeQt Core Widgets OpenGL OpenGLWidgets)
if (COVISE_USE_X11)
    if (Qt5X11Extras_FOUND)
        qt_use_modules(WindowTypeQt X11Extras)
        add_definitions(-DHAVE_QTX11EXTRAS)
    endif()

    if (X11_FOUND)
        target_link_libraries(WindowTypeQt ${X11_X11_LIB})
        if (X11_ICE_FOUND)
            target_link_libraries(WindowTypeQt ${X11_ICE_LIB})
        endif()
    endif()
endif()
