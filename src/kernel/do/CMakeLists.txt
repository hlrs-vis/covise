# @file
# 
# CMakeLists.txt for kernel - do
#
# @author Blasius Czink

ADD_DEFINITIONS(-DDO_EXPORTS)

SET(DO_SOURCES
  covise_gridmethods.cpp
  covise_statics.cpp
  coDoColormap.cpp
  coDoGeometry.cpp
  coDistributedObject.cpp
  coDoUnstructuredGrid.cpp
  coDoSet.cpp
  coDoData.cpp
  coDoPoints.cpp
  coDoSpheres.cpp
  coDoLines.cpp
  coDoPolygons.cpp
  coDoTriangleStrips.cpp
  coDoStructuredGrid.cpp
  coDoRectilinearGrid.cpp
  coDoUniformGrid.cpp
  coDoIntArr.cpp
  coDoText.cpp
  coDoTexture.cpp
  coDoPixelImage.cpp
  coDoBasisTree.cpp
  coDoOctTree.cpp
  coDoOctTreeP.cpp
  coShmPtrArray.cpp
  coDoDoubleArr.cpp
)

SET(DO_HEADERS
  Triangulate.h
  covise_gridmethods.h
  coDoAbstractStructuredGrid.h
  coDoColormap.h
  coDoGeometry.h
  coDistributedObject.h
  coDoUnstructuredGrid.h
  coDoSet.h
  coDoData.h
  coDoCoordinates.h
  coDoPoints.h
  coDoSpheres.h
  coDoLines.h
  coDoPolygons.h
  coDoTriangleStrips.h
  coDoStructuredGrid.h
  coDoRectilinearGrid.h
  coDoUniformGrid.h
  coDoIntArr.h
  coDoText.h
  coDoTexture.h
  coDoPixelImage.h
  coDoBasisTree.h
  coDoOctTree.h
  coDoOctTreeP.h
  coShmPtrArray.h
  coDoDoubleArr.h
)

ADD_COVISE_LIBRARY(coDo ${COVISE_LIB_TYPE} ${DO_SOURCES} ${DO_HEADERS})
TARGET_LINK_LIBRARIES(coDo coCore coNet coConfig)

COVISE_INSTALL_TARGET(coDo)
COVISE_INSTALL_HEADERS(do ${DO_HEADERS})
