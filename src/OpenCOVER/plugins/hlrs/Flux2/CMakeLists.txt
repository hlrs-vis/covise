USING(Pthreads)
using(VRML)

if(APPLE)
   return()
endif()

SET(HEADERS
  Flux2.h
)
SET(SOURCES
  Flux2.cpp
)
cover_add_plugin(Flux2)
TARGET_LINK_LIBRARIES(Flux2 ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Flux2)
