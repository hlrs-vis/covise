SET(HEADERS
)
SET(SOURCES
  ReadIhs.cpp
)
covise_add_module(IO ReadIhs ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(ReadIhs coApi coAppl coCore)
