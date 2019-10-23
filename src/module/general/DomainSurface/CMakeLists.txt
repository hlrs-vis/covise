
SET(HEADERS
  
)

SET(SOURCES
  DomainSurface.cpp
)

SET(EXTRASOURCES
  DomainSurface.h
)

ADD_COVISE_MODULE(Mapper DomainSurface ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(DomainSurface  coApi coAppl coCore )

COVISE_INSTALL_TARGET(DomainSurface)
