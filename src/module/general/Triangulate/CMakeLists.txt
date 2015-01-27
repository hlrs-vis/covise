
ADD_DEFINITIONS(-DTETLIBRARY)

SET(HEADERS
  
)

SET(SOURCES
  Triangulate.cpp
  dc.cpp
  memory.cpp
  sort.cpp
  edge.cpp
  tetgen.cpp
  predicates.cpp
)

SET(EXTRASOURCES
  Triangulate.h
  dc.h
  defs.h
  edge.h
  tetgen.h
)

ADD_COVISE_MODULE(Filter Triangulate ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Triangulate  coApi coAppl coCore )

COVISE_WNOERROR(Triangulate)

COVISE_INSTALL_TARGET(Triangulate)
