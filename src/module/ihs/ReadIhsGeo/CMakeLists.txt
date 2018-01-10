SET(HEADERS
)
SET(SOURCES
  ReadIhsGeo.cpp
)
covise_add_module(IO ReadIhsGeo ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(ReadIhsGeo coApi coAppl coCore)
covise_install_target(ReadIhsGeo)
