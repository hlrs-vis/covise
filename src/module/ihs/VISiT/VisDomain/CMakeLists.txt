SET(HEADERS
)
SET(SOURCES
  VisDomain.cpp
)
covise_add_module(VISiT VisDomain ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(VisDomain coApi coAppl coCore)
