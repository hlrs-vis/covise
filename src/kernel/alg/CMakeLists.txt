# @file
# 
# CMakeLists.txt for kernel - alg
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_ALG)

SET(ALG_SOURCES
  DeleteUnusedPoints.cpp
  coUniTracer.cpp
  coSphere.cpp
  coVectField.cpp
  coChemicalElement.cpp
  coBoundingBox.cpp
)

SET(ALG_HEADERS
  CuttingTables.h
  coUniTracer.h
  IsoCuttingTables.h
  DeleteUnusedPoints.h
  coSphere.h
  coChemicalElement.h
  coVectField.h
  CuttingSurfaceGPMUtil.h
  IsoSurfaceGPMUtil.h
  coBoundingBox.h
  unordered_set.h
)
SET(ALG_SOURCES ${ALG_SOURCES}
  coColors.cpp
  coComplexModules.cpp
  coCuttingSurface.cpp
  coIsoSurface.cpp
  MagmaUtils.cpp
  coFeatureLines.cpp
  coMiniGrid.cpp
  coCellToVert.cpp
  coFixUsg.cpp
)
SET(ALG_HEADERS ${ALG_HEADERS}
  coColors.h
  coComplexModules.h
  coCuttingSurface.h
  CuttingTables.h
  RainAlgorithm.h
  IsoCuttingTables.h
  coIsoSurface.h
  MagmaUtils.h
  coFeatureLines.h
  coMiniGrid.h
  coCellToVert.h
  coFixUsg.h
)

USING(VTK optional)
IF(VTK_FOUND)
   # add additional sources
   LIST(APPEND ALG_SOURCES coVtk.cpp)
   LIST(APPEND ALG_HEADERS coVtk.h)
ENDIF()


ADD_COVISE_LIBRARY(coAlg ${COVISE_LIB_TYPE} ${ALG_SOURCES} ${ALG_HEADERS})
TARGET_LINK_LIBRARIES(coAlg coAppl coApi coCore coConfig ${EXTRA_LIBS})

IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_COVISE_COMPILE_FLAGS(coAlg "-Wno-uninitialized")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

COVISE_INSTALL_TARGET(coAlg)
