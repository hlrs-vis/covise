
INCLUDE_DIRECTORIES(
  ../CellToVert
)

SET(HEADERS
  
)

SET(SOURCES
  CutGeometry.cpp
)

SET(EXTRASOURCES
  CutGeometry.h
)

ADD_COVISE_MODULE(Filter CutGeometry ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(CutGeometry  coAppl coApi coCore)

COVISE_INSTALL_TARGET(CutGeometry)
