USING(XERCESC)
USING(OsgCal)
using(VRML)
using(GDAL)

include_directories(../../../DrivingSim)
IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()


covise_find_package(OpenSceneGraph 3.2.0 COMPONENTS osgTerrain)
COVISE_FIND_PACKAGE(osgTerrain)
COVISE_FIND_PACKAGE(V8)

if(NOT V8_FOUND)
    return()
endif()

set(HEADERS
  OpenScenarioPlugin.h
  ScenarioManager.h
  Maneuver.h
  Act.h
  Entity.h
)
set(SOURCES
  OpenScenarioPlugin.cpp
  ScenarioManager.cpp
  Maneuver.cpp
  Act.cpp
  Entity.cpp
)

SET(TS_HEADERS
  ../TrafficSimulation/Vehicle.h
  ../TrafficSimulation/DonkeyCar.h
  ../TrafficSimulation/VehicleGeometry.h
  ../TrafficSimulation/CarGeometry.h
  ../TrafficSimulation/VehicleManager.h
  ../TrafficSimulation/VehicleFactory.h
  ../TrafficSimulation/HumanVehicle.h
  ../TrafficSimulation/HumanVehicleGeometry.h
  ../TrafficSimulation/AgentVehicle.h
  ../TrafficSimulation/VehicleUtils.h
  ../TrafficSimulation/mtrand.h
  ../TrafficSimulation/projects/radarcones.hpp
  ../TrafficSimulation/PorscheFFZ.h
  ../TrafficSimulation/UDPBroadcast.h
  ../TrafficSimulation/PedestrianManager.h
  ../TrafficSimulation/Pedestrian.h
  ../TrafficSimulation/PedestrianGeometry.h
  ../TrafficSimulation/PedestrianFactory.h
  ../TrafficSimulation/TrafficSimulation.h
)
SET(TS_SOURCES
  ../TrafficSimulation/Vehicle.cpp
  ../TrafficSimulation/DonkeyCar.cpp
  ../TrafficSimulation/CarGeometry.cpp
  ../TrafficSimulation/VehicleManager.cpp
  ../TrafficSimulation/VehicleFactory.cpp
  ../TrafficSimulation/HumanVehicle.cpp
  ../TrafficSimulation/HumanVehicleGeometry.cpp
  ../TrafficSimulation/AgentVehicle.cpp
  ../TrafficSimulation/VehicleUtils.cpp
  ../TrafficSimulation/mtrand.cpp
  ../TrafficSimulation/projects/radarcones.cpp
  ../TrafficSimulation/PorscheFFZ.cpp
  ../TrafficSimulation/UDPBroadcast.cpp
  ../TrafficSimulation/PedestrianManager.cpp
  ../TrafficSimulation/Pedestrian.cpp
  ../TrafficSimulation/PedestrianGeometry.cpp
  ../TrafficSimulation/PedestrianFactory.cpp
  ../TrafficSimulation/TrafficSimulation.cpp
)


INCLUDE_DIRECTORIES(
   ../../../DrivingSim/VehicleUtil
   ../TrafficSimulation
   ${OSGTERRAIN_INCLUDE_DIR}
   ${V8_INCLUDE_DIR}
)
# build the plugin "OpenScenarioPlugin" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(OpenScenarioPlugin ${HEADERS} ${SOURCES} ${TS_HEADERS} ${TS_SOURCES} coOpenVehicleUtil coOpenScenario ${OSGTERRAIN_LIBRARIES} ${EXTRA_LIBS} RoadTerrain SteeringWheel)
