SET(HEADERS
  Hello.h
)
SET(SOURCES
  Hello.cpp
)
covise_add_module(Examples Hello ${EXTRASOURCES} ${SOURCES} ${HEADERS})
