SET(HEADERS
  ReadObjSimple.h
)
SET(SOURCES
  ReadObjSimple.cpp
)
covise_add_module(Examples ReadObjSimple ${EXTRASOURCES} ${SOURCES} ${HEADERS})
