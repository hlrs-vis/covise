using(Boost)

SET(HEADERS
  ReadVois.h
  VoisGlobal.h
)
SET(SOURCES
  ReadVois.cpp
)
#covise_add_module(IO ReadVois ${EXTRASOURCES} ${SOURCES} ${HEADERS})
ADD_COVISE_MODULE(IO ReadVois ${EXTRASOURCES} ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES(ReadVois coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadVois)

