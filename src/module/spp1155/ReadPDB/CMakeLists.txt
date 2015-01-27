using(VTK)

if (${VTK_MAJOR_VERSION} GREATER 5)
   return()
endif()

SET(HEADERS
)
SET(SOURCES
  ReadPDB.cpp
  vtkMyPDBReader.cxx
  vtkMyMoleculeReaderBase.cxx
)
ADD_COVISE_MODULE(IO ReadPDB ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadPDB coApi coAppl coCore coAlg)
COVISE_INSTALL_TARGET(ReadPDB)
