SET(HEADERS
  ParamTest.h
)
SET(SOURCES
  ParamTest.cpp
)
covise_add_module(Examples ParamTest ${EXTRASOURCES} ${SOURCES} ${HEADERS})
