SET(HEADERS
)
SET(SOURCES
  ReadIhsGrid.cpp
)
covise_add_module(IO ReadIhsGrid ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(ReadIhsGrid coApi coAppl coCore)
covise_install_target(ReadIhsGrid)
