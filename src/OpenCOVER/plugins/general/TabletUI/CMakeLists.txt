if(NOT COVISE_USE_QT)
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
    TabletUIPlugin.h
)

set(SOURCES
    TabletUIPlugin.cpp
)

cover_add_plugin(TabletUI ${HEADERS} ${SOURCES})
qt_use_modules(TabletUI Core Widgets)
target_link_libraries(TabletUI coTUI)
if (COVISE_USE_X11)
    if (Qt5X11Extras_FOUND)
        qt_use_modules(TabletUI X11Extras)
        add_definitions(-DHAVE_QTX11EXTRAS)
    endif()

    if (X11_FOUND)
        target_link_libraries(TabletUI ${X11_X11_LIB})
        if (X11_ICE_FOUND)
            target_link_libraries(TabletUI ${X11_ICE_LIB})
        endif()
    endif()
endif()
