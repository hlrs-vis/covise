USING(Pthreads)
using(VRML)

if(APPLE)
   return()
endif()

SET(HEADERS
  Skateboard.h
)
SET(SOURCES
  Skateboard.cpp
)
cover_add_plugin(Skateboard)
TARGET_LINK_LIBRARIES(Skateboard ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Skateboard)
