SET(HEADERS
  TestUIF.h
)
SET(SOURCES
  TestUIF.cpp
)
covise_add_module(Examples TestUIF ${EXTRASOURCES} ${SOURCES} ${HEADERS})
