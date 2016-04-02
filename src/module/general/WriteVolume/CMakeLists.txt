USING(Virvo:fileio)

SET(HEADERS
  
)

SET(SOURCES
  WriteVolume.cpp
)

SET(EXTRASOURCES
  WriteVolume.h
)

ADD_COVISE_MODULE(IO WriteVolume ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(WriteVolume  coApi coAppl coCore ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(WriteVolume)
