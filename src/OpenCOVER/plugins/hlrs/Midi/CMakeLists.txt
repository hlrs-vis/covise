using(MIDIFILE)
using(SDL2)
using(FFTW)
using(VRML)
#USE_MIDIFILE()
#    MESSAGE(${MIDIFILE_INCLUDE_DIR})
#    INCLUDE_DIRECTORIES(${MIDIFILE_INCLUDE_DIR})
SET(HEADERS
  midiPlugin.h
  open62541.h
  pubFunctions.h
  server.h
)
SET(SOURCES
  midiPlugin.cpp
  #open62541.cpp
  #pubFunctions.cpp
  
)
cover_add_plugin(Midi)
IF(WIN32)
 TARGET_LINK_LIBRARIES(Midi winmm ${EXTRA_LIBS})
ENDIF(WIN32)
#COVER_INSTALL_PLUGIN(MIDI)
