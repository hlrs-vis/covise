SET(HEADERS
  ReadElmer.h
)
SET(SOURCES
  ReadElmer.cpp
)
covise_add_module(IO ReadElmer ${EXTRASOURCES} ${SOURCES} ${HEADERS})
