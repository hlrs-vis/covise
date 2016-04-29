ADD_DEFINITIONS("-DGATE -DCONTOURS")

SET(HEADERS
  ShowDistBoco.h
)
SET(SOURCES
  ShowDistBoco.cpp
)
covise_add_module(VISiT ShowDistBoco ${EXTRASOURCES} ${HEADERS} ${SOURCES})
target_link_libraries(ShowDistBoco coApi coAppl coCore)
