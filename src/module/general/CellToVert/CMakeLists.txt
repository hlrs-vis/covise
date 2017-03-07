
SET(HEADERS
  
)

SET(SOURCES
  CellToVert.cpp
)

SET(EXTRASOURCES
  CellToVert.h
)

ADD_COVISE_MODULE(Interpolator CellToVert ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(CellToVert  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(CellToVert)
