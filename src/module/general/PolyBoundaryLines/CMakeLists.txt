
SET(HEADERS
  
)

SET(SOURCES
  PolyBoundaryLines.cpp
)

SET(EXTRASOURCES
  PolyBoundaryLines.h
)

USE_VTK(OPTIONAL)
ADD_COVISE_MODULE(Filter  PolyBoundaryLines ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(PolyBoundaryLines  coAlg coApi coAppl coCore )
COVISE_WNOERROR(PolyBoundaryLines)

COVISE_INSTALL_TARGET(PolyBoundaryLines)
