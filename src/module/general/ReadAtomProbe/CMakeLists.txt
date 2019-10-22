SET(HEADERS
  ReadAtomProbe.h
)
SET(SOURCES
  ReadAtomProbe.cpp
)

ADD_COVISE_MODULE(IO ReadAtomProbe ${EXTRASOURCES})
covise_wnoerror(ReadAtomProbe)
TARGET_LINK_LIBRARIES(ReadAtomProbe coApi coAppl coCore coAlg)
COVISE_INSTALL_TARGET(ReadAtomProbe)
