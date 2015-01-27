
# possiblyaliasing

#ADD_DEFINITIONS()
#INCLUDE_DIRECTORIES()

SET(HEADERS
  
)

SET(SOURCES
  Map1D.cpp
  Node.cpp
  ReadANSYS.cpp
  ReadANSYS1.cpp
  ReadANSYS2.cpp
  ReadRST.cpp
  ReadRSTDerived.cpp
  ANSYS.cpp
)

SET(EXTRASOURCES
  BinHeader.h
  DOFData.h
  DOFOptions.h
  DerivedData.h
  EType.h
  Element.h
  GeometryHeader.h
  Map1D.h
  Node.h
  ReadANSYS.h
  ReadRST.h
  RstHeader.h
  SolutionHeader.h
  ANSYS.h
)

USE_VTK(OPTIONAL)
ADD_COVISE_MODULE(IO ReadANSYS ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadANSYS  coAlg coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(ReadANSYS)
