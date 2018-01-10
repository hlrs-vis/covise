set(HEADERS
  ReadVTU.h
)
set(SOURCES
  ReadVTU.cpp 
)

covise_add_module(Test ReadVTU ${HEADERS} ${SOURCES})
covise_wnoerror(ReadVTU)
TARGET_LINK_LIBRARIES(ReadVTU coApi coAppl coCore ${EXTRA_LIBS})
covise_install_target(ReadVTU)
