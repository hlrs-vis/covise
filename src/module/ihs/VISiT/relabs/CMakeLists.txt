INCLUDE_DIRECTORIES(
../lib
)

SET(HEADERS
  relabs.h
)
SET(SOURCES
  relabs.cpp
)

covise_add_module(VISiT relabs ${EXTRASOURCES} ${HEADERS} ${SOURCES})
target_link_libraries(relabs coApi coAppl coCore)
covise_install_target(relabs)
