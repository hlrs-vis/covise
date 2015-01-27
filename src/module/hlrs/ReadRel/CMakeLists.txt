SET(HEADERS
)
SET(SOURCES
  ReadRel.cpp
  reldata.cpp
  usefullroutines.cpp
  writetext.cpp
)
ADD_COVISE_MODULE(IO ReadRel ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadRel coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadRel)
