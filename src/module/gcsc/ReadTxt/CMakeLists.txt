set(HEADERS
  ReadTxt.h
)
set(SOURCES
  ReadTxt.cpp 
)

covise_add_module(Test TxtReader ${HEADERS} ${SOURCES})
