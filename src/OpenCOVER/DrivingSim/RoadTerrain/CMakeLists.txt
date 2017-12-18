if(NOT COVISE_BUILD_DRIVINGSIM)
  return()
endif()

using(VehicleUtil)

covise_find_package(OpenSceneGraph 3.2.0 COMPONENTS osgTerrain)
COVISE_FIND_PACKAGE(osgTerrain)
IF(NOT OSGTERRAIN_FOUND)
  RETURN()
ENDIF()

covise_find_package(GDAL)
if(NOT GDAL_FOUND)
    return()
endif()

set(HEADERS
    DecoratedGeometryTechnique.h
    LevelTerrainCallback.h
    RoadTerrainLoader.h
    mtrand.h
)

set(SOURCES
    DecoratedGeometryTechnique.cpp
    LevelTerrainCallback.cpp
    RoadTerrainLoader.cpp
    mtrand.cpp
)

include_directories(
    ${OPENSCENEGRAPH_INCLUDE_DIRS}
    ${GDAL_INCLUDE_DIR}
)

add_covise_library(coRoadTerrain SHARED ${SOURCES} ${HEADERS})

target_link_libraries(coRoadTerrain
     ${OPENSCENEGRAPH_LIBRARIES}
     ${GDAL_LIBRARIES}
     coOpenCOVER
     ${EXTRA_LIBS}
   )

covise_install_target(coRoadTerrain)
covise_wnoerror(coRoadTerrain)
