set(HEADERS
  ReadVTU.h
)
set(SOURCES
  ReadVTU.cpp 
)

covise_add_module(Test ReadVTU ${HEADERS} ${SOURCES})
