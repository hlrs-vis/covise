
SET(HEADERS
  ReadCUBE.h
)
SET(SOURCES
  ReadCUBE.cpp
)
covise_add_module(IO ReadCUBE ${HEADERS} ${SOURCES})
covise_install_target(ReadCUBE)
