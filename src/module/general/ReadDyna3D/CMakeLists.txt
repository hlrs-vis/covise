
SET(HEADERS
  Element.h
  ReadDyna3D.h
)

SET(SOURCES
  Element.cpp
  ReadDyna3D.cpp
)

SET(EXTRASOURCES
  Element.h
  ReadDyna3D.h
)

ADD_COVISE_MODULE(IO ReadDyna3D ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadDyna3D  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(ReadDyna3D)
