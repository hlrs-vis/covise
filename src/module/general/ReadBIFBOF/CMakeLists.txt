USING(BIFBOF)

SET(HEADERS
  
)

SET(SOURCES
  ReadBIFBOF.cpp
  intel.cpp
  BofVectorData.cpp
  BofScalarData.cpp
  BifTriangConns.cpp
  BifTetraConns.cpp
  BifQuadConns.cpp
  BifPyramidConns.cpp
  BifPentaConns.cpp
  BifNodalPoints.cpp
  BifGeoElements.cpp
  BifElement.cpp
  BifCubeConns.cpp
  BifBofInterface.cpp
)

SET(EXTRASOURCES
  ReadBIFBOF.h
  BofVectorData.h
  BofScalarData.h
  BifTriangConns.h
  BifTetraConns.h
  BifQuadConns.h
  BifPyramidConns.h
  BifPentaConns.h
  BifNodalPoints.h
  BifGeoElements.h
  BifElement.h
  BifCubeConns.h
  BifBofInterface.h
)

ADD_COVISE_MODULE(IO ReadBIFBOF ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadBIFBOF  coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadBIFBOF)
