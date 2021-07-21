using(VRML)


SET(HEADERS
  Wheelchair.h
)
SET(SOURCES
  Wheelchair.cpp
)
cover_add_plugin(Wheelchair)
TARGET_LINK_LIBRARIES(Wheelchair ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Wheelchair)
