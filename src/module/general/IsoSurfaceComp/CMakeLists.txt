# CONFIG *= wnouninitialized # because of IsoSurfaceGPMUtil.h

ADD_DEFINITIONS(-D_COMPLEX_MODULE_)

INCLUDE_DIRECTORIES(
  ../IsoSurface
)

SET(HEADERS
  
)

SET(SOURCES
  ../IsoSurface/IsoPoint.cpp
  ../IsoSurface/IsoSurface.cpp
)

SET(EXTRASOURCES
  ../IsoSurface/IsoPoint.h
  ../IsoSurface/IsoSurface.h
)

USE_VTK(OPTIONAL)
ADD_COVISE_MODULE(Mapper IsoSurfaceComp ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(IsoSurfaceComp  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(IsoSurfaceComp)
