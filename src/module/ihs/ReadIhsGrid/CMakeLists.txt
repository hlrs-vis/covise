SET(HEADERS
)
SET(SOURCES
  ReadIhsGrid.cpp
)
covise_add_module(IO_Module ReadIhsGrid ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(ReadIhsGrid coApi coAppl coCore)
