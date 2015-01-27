
# CONFIG *= wnouninitialized # because of IsoSurfaceGPMUtil.h

SET(HEADERS
  IsoPoint.h
  IsoSurface.h
)

SET(SOURCES
  IsoPoint.cpp
  IsoSurface.cpp
)

SET(EXTRASOURCES
  IsoPoint.h
  IsoSurface.h
)

USE_VTK(OPTIONAL)
ADD_COVISE_MODULE(Mapper IsoSurface )
TARGET_LINK_LIBRARIES(IsoSurface  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(IsoSurface)
