SET(HEADERS
  ReadCSVPoly.h
)
SET(SOURCES
  ReadCSVPoly.cpp
)
covise_add_module(IO ReadCSVPoly ${EXTRASOURCES} ${SOURCES} ${HEADERS})
covise_install_target(ReadCSVPoly)
