
SET(HEADERS
  ReadCalculix.h
)

SET(SOURCES
  ReadCalculix.cpp
)


ADD_COVISE_MODULE(Stellba ReadCalculix ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadCalculix coApi coAppl coCore ${EXTRA_LIBS})
COVISE_WNOERROR(ReadCalculix)

COVISE_INSTALL_TARGET(ReadCalculix)
