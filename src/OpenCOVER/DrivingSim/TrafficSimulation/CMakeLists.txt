IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(XERCESC OSGCAL CAL3D)
using(VRML)
using(V8 optional)
using(VehicleUtil)
using(RoadTerrain)

covise_find_package(OpenSceneGraph 3.2.0 QUIET COMPONENTS osgTerrain)
COVISE_FIND_PACKAGE(osgTerrain)
COVISE_FIND_PACKAGE(GDAL)

IF(NOT OSGCAL_FOUND)
  RETURN()
ENDIF()

if(NOT GDAL_FOUND)
    return()
endif()

SET(HEADERS
  AgentVehicle.h
  CarGeometry.h
  coEntity.h
  coTrafficSimulation.h
  DonkeyCar.h
  HumanVehicle.h
  HumanVehicleGeometry.h
  ImageDescription.h
  Pedestrian.h
  PedestrianFactory.h
  PedestrianGeometry.h
  PedestrianManager.h
  PorscheFFZ.h
  projects/radarcones.hpp
  TrafficSimulation.h
  UDPBroadcast.h
  Vehicle.h
  VehicleFactory.h
  VehicleGeometry.h
  VehicleManager.h
  VehicleUtils.h
)
SET(SOURCES
  AgentVehicle.cpp
  CarGeometry.cpp
  coEntity.cpp
  coTrafficSimulation.cpp
  DonkeyCar.cpp
  HumanVehicle.cpp
  HumanVehicleGeometry.cpp
  ImageDescription.cpp
  Pedestrian.cpp
  PedestrianFactory.cpp
  PedestrianGeometry.cpp
  PedestrianManager.cpp
  PorscheFFZ.cpp
  projects/radarcones.cpp
  TrafficSimulation.cpp
  UDPBroadcast.cpp
  Vehicle.cpp
  VehicleFactory.cpp
  VehicleManager.cpp
  VehicleUtils.cpp
)


INCLUDE_DIRECTORIES(
   ${OSGTERRAIN_INCLUDE_DIR}
   ${OSGCAL3D_INCLUDE_DIR}
   ${CAL3D_INCLUDE_DIR}
   ${GDAL_INCLUDE_DIR}
)


ADD_COVISE_LIBRARY(coTrafficSimulation SHARED ${SOURCES} ${HEADERS})
target_compile_definitions(coTrafficSimulation PRIVATE COVER_PLUGIN_NAME="coTrafficSimulation")
qt_use_modules(coTrafficSimulation Script ScriptTools)

COVISE_WNOERROR(coTrafficSimulation)

target_link_libraries(coTrafficSimulation
 ${OSGTERRAIN_LIBRARIES}
 ${OSGCAL_LIBRARIES}
 ${OPENSCENEGRAPH_LIBRARIES}
 ${EXTRA_LIBS}
 #SteeringWheel #required for vrmlNodeVehicle (windows)
 coOpenVehicleUtil)
 

COVISE_INSTALL_TARGET(coTrafficSimulation)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(coTrafficSimulation "/Zc:wchar_t-")
ENDIF()


