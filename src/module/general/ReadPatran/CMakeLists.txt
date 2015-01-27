
SET(HEADERS
  
)

SET(SOURCES
  AscStream.cpp
  ElementAsciiFile.cpp
  ElementFile.cpp
  NeutralFile.cpp
  NodalFile.cpp
  ReadPatran.cpp
  StepFile.cpp
  istreamFTN.cpp
)

SET(EXTRASOURCES
  AscStream.h
  ElementAsciiFile.h
  ElementFile.h
  NeutralFile.h
  NodalFile.h
  ReadPatran.h
  StepFile.h
  istreamFTN.h
)

ADD_COVISE_MODULE(IO ReadPatran ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadPatran  coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadPatran)
