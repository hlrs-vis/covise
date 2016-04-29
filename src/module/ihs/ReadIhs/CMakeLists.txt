SET(HEADERS
)
SET(SOURCES
  ReadIhs.cpp
)
covise_add_module(IO_Module ReadIhs ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(ReadIhs coApi coAppl coCore)
