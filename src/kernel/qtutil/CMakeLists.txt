if(NOT COVISE_USE_QT)
    return()
endif()

set(QTUTIL_SOURCES
    FileSysAccess.cpp
    NetHelp.cpp
    NonBlockingDialogue.cpp
)

set(QTUTIL_HEADERS
    FileSysAccess.h
    NetHelp.h
    NonBlockingDialogue.h
    Qt5_15_deprecated.h
)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_covise_library(coQtUtil ${COVISE_LIB_TYPE} ${QTUTIL_SOURCES} ${QTUTIL_HEADERS} ${QTUTIL_MOC_HEADERS} ${QTUTIL_MOC_SRCS})
target_link_libraries(coQtUtil coUtil coNet coConfig)
qt_use_modules(coQtUtil Core Network Widgets)
covise_install_target(coQtUtil)
covise_install_headers(qtutil ${QTUTIL_HEADERS})
