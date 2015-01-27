USING(CGNS)

SET(HEADERS
  ReadCGNS.h
  common.h
  zone.h
)

SET(SOURCES
  ReadCGNS.cpp
  zone.cpp
)

SET(EXTRASOURCES
  ReadCGNS.h
  common.h
  zone.h
)

ADD_COVISE_MODULE(IO ReadCGNS-spbgpu ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadCGNS-spbgpu coApi coAppl coCore coUtil ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(ReadCGNS-spbgpu)
