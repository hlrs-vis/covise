IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

COVISE_FIND_PACKAGE(OSGCAL)
COVISE_FIND_PACKAGE(CAL3D)

USING(XERCESC OSGCAL CAL3D)
using(VRML)
using(TrafficSimulation)
using(V8 optional)

IF(NOT OSGCAL_FOUND)
  RETURN()
ENDIF()

covise_find_package(OpenSceneGraph 3.2.0 COMPONENTS osgTerrain)
COVISE_FIND_PACKAGE(osgTerrain)
COVISE_FIND_PACKAGE(GDAL)

if(NOT GDAL_FOUND)
    return()
endif()

SET(HEADERS
  TrafficSimulationPlugin.h
)
SET(SOURCES
  TrafficSimulationPlugin.cpp
)


INCLUDE_DIRECTORIES(
   ../../../DrivingSim/VehicleUtil
   ../../../DrivingSim/coTrafficSimulation
   ${OSGTERRAIN_INCLUDE_DIR}
   ${OSGCAL3D_INCLUDE_DIR}
   ${CAL3D_INCLUDE_DIR}
   ${GDAL_INCLUDE_DIR}
)

cover_add_plugin(TrafficSimulation  ${HEADERS} ${SOURCES})
qt_use_modules(TrafficSimulation Script ScriptTools)
target_link_libraries(TrafficSimulation
 ${EXTRA_LIBS}
 SteeringWheel #required for vrmlNodeVehicle (windows)
 )

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(TrafficSimulation "/Zc:wchar_t-")
ENDIF()

COVISE_WNOERROR(TrafficSimulation)

#COVER_INSTALL_PLUGIN(TrafficSimulation)
