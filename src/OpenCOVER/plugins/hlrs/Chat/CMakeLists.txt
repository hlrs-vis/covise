using(Speex)


    
SET(HEADERS
  chatPlugin.h
)
SET(SOURCES
  chatPlugin.cpp
)
cover_add_plugin(Chat)

if(Qt5Multimedia_FOUND)
  qt_use_modules(Chat Multimedia)
else(Qt5Multimedia_FOUND)
return()
endif(Qt5Multimedia_FOUND)
IF(WIN32)
 TARGET_LINK_LIBRARIES(Chat winmm ${EXTRA_LIBS})
ENDIF(WIN32)
#COVER_INSTALL_PLUGIN(MIDI)
