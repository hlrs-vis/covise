SET(HEADERS
)
SET(SOURCES
  VeloIHS.cpp
)
covise_add_module(VISiT VeloIHS ${EXTRASOURCES} ${HEADERS} ${SOURCES})
target_link_libraries(VeloIHS coApi coAppl coCore)
covise_install_target(VeloIHS)
