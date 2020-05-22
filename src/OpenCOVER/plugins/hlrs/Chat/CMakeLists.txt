using(Speex)

set(HEADERS
    chatPlugin.h
    )
set(SOURCES
    chatPlugin.cpp
    )

if(Qt5Multimedia_FOUND)
    cover_add_plugin(Chat)
    qt_use_modules(Chat Multimedia)
else(Qt5Multimedia_FOUND)
    return()
endif(Qt5Multimedia_FOUND)
if(WIN32)
    target_link_libraries(Chat winmm ${EXTRA_LIBS})
endif(WIN32)
#cover_install_plugin(Chat)
