
SET(HEADERS
  
)

SET(SOURCES
  SmoothSurface.cpp
  Surface.cpp
)

SET(EXTRASOURCES
  SmoothSurface.h
  Surface.h
)

ADD_COVISE_MODULE(Filter SmoothSurface ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(SmoothSurface  coApi coAppl coCore )

COVISE_INSTALL_TARGET(SmoothSurface)
