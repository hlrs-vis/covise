
SET(HEADERS
  
)

SET(SOURCES
  DomainSurface.cpp
)

SET(EXTRASOURCES
  DomainSurface.h
)

ADD_COVISE_MODULE(Filter DomainSurface ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(DomainSurface  coApi coAppl coCore )

COVISE_INSTALL_TARGET(DomainSurface)
