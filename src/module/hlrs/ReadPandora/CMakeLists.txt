using(HDF5:hl)

set(HEADERS
   ReadPandora.h
)
set(SOURCES
   ReadPandora.cpp
)

covise_add_module(IO ReadPandora ${HEADERS} ${SOURCES})
TARGET_LINK_LIBRARIES(ReadPandora coReader ${EXTRA_LIBS})
covise_install_target(ReadPandora)
