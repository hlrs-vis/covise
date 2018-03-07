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
  Condition.h
  Act.h
  Entity.h
  myFactory.h
  FollowTrajectory.h
  Trajectory.h
  RouteInfo.h
  CameraSensor.h
  Position.h
  Spline.h
  ReferencePosition.h
)
set(SOURCES
  OpenScenarioPlugin.cpp
  ScenarioManager.cpp
  Maneuver.cpp
  Act.cpp
  Entity.cpp
  Condition.cpp
  myFactory.cpp
  FollowTrajectory.cpp
  Trajectory.cpp
  RouteInfo.cpp
  CameraSensor.cpp
  Position.cpp
  Spline.cpp
  ReferencePosition.cpp
)


INCLUDE_DIRECTORIES(
   ${OSGTERRAIN_INCLUDE_DIR}
)
# build the plugin "OpenScenarioPlugin" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(OpenScenarioPlugin ${HEADERS} ${SOURCES} ${OSGTERRAIN_LIBRARIES} ${EXTRA_LIBS} coRoadTerrain)
covise_wnoerror(OpenScenarioPlugin)
set_target_properties(OpenScenarioPlugin PROPERTIES OUTPUT_NAME "OpenScenario")
