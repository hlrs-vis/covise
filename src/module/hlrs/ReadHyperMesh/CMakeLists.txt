SET(HEADERS
  ReadHyperMesh.h
)
SET(SOURCES
  ReadHyperMesh.cpp
)
ADD_COVISE_MODULE(IO ReadHyperMesh ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadHyperMesh coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadHyperMesh)
