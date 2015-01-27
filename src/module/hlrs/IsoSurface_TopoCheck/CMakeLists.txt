SET(HEADERS
  IsoSurface_TopoCheck.h
  marching_cubes_jgt/fparser.h
  marching_cubes_jgt/LookUpTable.h
  marching_cubes_jgt/MarchingCubes.h
  marching_cubes_jgt/fparser.h
)
SET(SOURCES
  IsoSurface_TopoCheck.cpp
  marching_cubes_jgt/fparser.cpp
  marching_cubes_jgt/main.cpp
  marching_cubes_jgt/LookUpTableTest.cpp
  marching_cubes_jgt/MarchingCubes.cpp
)
ADD_COVISE_MODULE(Mapper IsoSurface_TopoCheck ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(IsoSurface_TopoCheck coApi coAppl coCore)
COVISE_INSTALL_TARGET(IsoSurface_TopoCheck)
