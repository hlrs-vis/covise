set(QTUTIL_SOURCES
    FileSysAccess.cpp
    NetHelp.cpp
)

set(QTUTIL_HEADERS
    FileSysAccess.h
    NetHelp.h
)

qt_wrap_cpp(WSLIB_MOC_SRCS ${QTUTIL_MOC_HEADERS})

add_covise_library(coQtUtil ${COVISE_LIB_TYPE} ${QTUTIL_SOURCES} ${QTUTIL_HEADERS} ${QTUTIL_MOC_HEADERS} ${QTUTIL_MOC_SRCS})
target_link_libraries(coQtUtil coUtil coNet coConfig)
qt_use_modules(coQtUtil Core Network)
covise_install_target(coQtUtil)
covise_install_headers(qtutil ${QTUTIL_HEADERS})
