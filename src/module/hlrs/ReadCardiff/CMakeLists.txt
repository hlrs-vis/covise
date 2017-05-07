SET(HEADERS
  ReadCardiff.h
)
SET(SOURCES
  ReadCardiff.cpp
)
ADD_COVISE_MODULE(IO ReadCardiff ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadCardiff coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadCardiff)
