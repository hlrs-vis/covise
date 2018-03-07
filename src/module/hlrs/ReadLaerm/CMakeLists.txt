
SET(HEADERS
  ReadLaerm.h
)
SET(SOURCES
  ReadLaerm.cpp
)


ADD_COVISE_MODULE(IO ReadLaerm ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadLaerm coReader coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadLaerm)