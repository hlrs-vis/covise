SET(HEADERS
  Snap2Surface.h
)

SET(SOURCES
  Snap2Surface.cpp
)

ADD_COVISE_MODULE(IO Snap2Surface ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Snap2Surface coApi coAppl coCore)
COVISE_INSTALL_TARGET(Snap2Surface)

