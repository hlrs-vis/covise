# __ autoconverted __
# PROJECT(SimplifySurface)

using(VTK)

# ADD_DEFINITIONS()
# INCLUDE_DIRECTORIES()

SET(HEADERS
  
)

SET(SOURCES
  Cholesky.cpp
  Edge.cpp
  EdgeCollapse.cpp
  EdgeCollapseBasis.cpp
  EdgeCollapseSimple.cpp
  EdgeContainer.cpp
  PQ.cpp
  Point.cpp
  SimplifySurfaceNT.cpp
  Triangle.cpp
  TriangleContainer.cpp
  Vertex.cpp
  VertexContainer.cpp
)

SET(EXTRASOURCES
  Edge.h
  EdgeCollapse.h
  EdgeCollapseBasis.h
  EdgeCollapseSimple.h
  EdgeContainer.h
  PQ.h
  Point.h
  SimplifySurfaceNT.h
  Triangle.h
  TriangleContainer.h
  Vertex.h
  VertexContainer.h
)

ADD_COVISE_MODULE(Filter SimplifySurface ${EXTRASOURCES} )
# old CONFIG: qt warn_on release incremental link_prl colib coalg coapi coappl math vtk_opt
# old LIBS: 
# old links: 
TARGET_LINK_LIBRARIES(SimplifySurface  coAlg coApi coAppl coCore ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(SimplifySurface)
