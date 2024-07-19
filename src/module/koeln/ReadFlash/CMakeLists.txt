using(HDF5:cxx_hl)
using(Boost)

SET(HEADERS
  
)

SET(SOURCES
  ReadFlash.cpp
)

SET(EXTRASOURCES
  ReadFlash.h
)

ADD_COVISE_MODULE(IO ReadFlash ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadFlash  coApi coAppl coCore ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(ReadFlash)
