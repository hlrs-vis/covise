SET(HEADERS
  ReadFconfig.h
)
SET(SOURCES
  ReadFconfig.cpp
)
ADD_COVISE_MODULE(IO ReadFconfig ${EXTRASOURCES})
covise_wnoerror(ReadFconfig)
TARGET_LINK_LIBRARIES(ReadFconfig coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadFconfig)
