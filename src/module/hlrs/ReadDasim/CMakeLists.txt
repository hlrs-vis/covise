SET(HEADERS
)
SET(SOURCES
  ReadDasim.cpp
)
ADD_COVISE_MODULE(IO ReadDasim ${EXTRASOURCES})
covise_wnoerror(ReadDasim)
TARGET_LINK_LIBRARIES(ReadDasim coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadDasim)
