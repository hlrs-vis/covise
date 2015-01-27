SET(HEADERS
  TestSoc.h
)
SET(SOURCES
  TestSoc.cpp
)
covise_add_module(Examples TestSoc ${EXTRASOURCES} ${SOURCES} ${HEADERS})
