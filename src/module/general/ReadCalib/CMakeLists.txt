
SET(HEADERS
  ReadCalib.h
)

SET(SOURCES
  ReadCalib.cpp
)

SET(EXTRASOURCES
  
)

ADD_COVISE_MODULE(IO ReadCalib )
TARGET_LINK_LIBRARIES(ReadCalib  coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadCalib)
