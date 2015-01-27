
SET(HEADERS
  Binary.h
  ReadVTF.h
  ReadVTFData.h
)

SET(SOURCES
  Binary.cpp
  ReadVTF.cpp
  ReadVTFData.cpp
)

SET(EXTRASOURCES
  Binary.h
  ReadVTF.h
  ReadVTFData.h
)

USE_VTK(OPTIONAL)
ADD_COVISE_MODULE(IO ReadVTF )
TARGET_LINK_LIBRARIES(ReadVTF  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadVTF)
