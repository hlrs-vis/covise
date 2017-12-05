USING(XERCESC)
USING(OsgCal)
using(VRML)
using(GDAL)
using(VehicleUtil)
using(TrafficSimulation)
using(OpenScenario)


covise_find_package(OpenSceneGraph 3.2.0 COMPONENTS osgTerrain)
COVISE_FIND_PACKAGE(osgTerrain)

set(HEADERS
  OpenScenarioPlugin.h
  ScenarioManager.h
  Maneuver.h
  Act.h
  Entity.h
  myFactory.h
  FollowTrajectory.h
  Trajectory.h
  CameraSensor.h
)
set(SOURCES
  OpenScenarioPlugin.cpp
  ScenarioManager.cpp
  Maneuver.cpp
  Act.cpp
  Entity.cpp
  myFactory.cpp
  FollowTrajectory.cpp
  Trajectory.cpp
  CameraSensor.cpp
)


INCLUDE_DIRECTORIES(
   ${OSGTERRAIN_INCLUDE_DIR}
)
# build the plugin "OpenScenarioPlugin" from the ingredients listed in HEADERS and SOURCES
<<<<<<< HEAD
cover_add_plugin(OpenScenarioPlugin ${HEADERS} ${SOURCES} ${OSGTERRAIN_LIBRARIES} ${EXTRA_LIBS} RoadTerrain SteeringWheel)
if(BASEARCHSUFFIX == cray64)
{
COVISE_WNOERROR(OpenScenarioPlugin)
}
=======
cover_add_plugin(OpenScenarioPlugin ${HEADERS} ${SOURCES} ${OSGTERRAIN_LIBRARIES} ${EXTRA_LIBS})
covise_wnoerror(OpenScenarioPlugin)
set_target_properties(OpenScenarioPlugin PROPERTIES OUTPUT_NAME "OpenScenario")
>>>>>>> c7be9f77e08addb0cae52532f4f14015d9934d6c
