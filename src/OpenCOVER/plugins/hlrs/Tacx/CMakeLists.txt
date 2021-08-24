USING(Pthreads)
using(VRML)
USING(LibUSB1)

if(APPLE)
   return()
endif()

SET(HEADERS
  TacxPlugin.h
  Tacx.h
)
SET(SOURCES
  TacxPlugin.cpp
  Tacx.cpp
)
cover_add_plugin(Tacx)
TARGET_LINK_LIBRARIES(Tacx ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Tacx)
