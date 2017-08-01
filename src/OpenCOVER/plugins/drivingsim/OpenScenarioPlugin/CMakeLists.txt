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
  myFactory.h
  FollowTrajectory.h
  Trajectory.h
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
)


INCLUDE_DIRECTORIES(
   ../../../DrivingSim/VehicleUtil
   ../../../DrivingSim/coTrafficSimulation
   ${OSGTERRAIN_INCLUDE_DIR}
   ${V8_INCLUDE_DIR}
)
# build the plugin "OpenScenarioPlugin" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(OpenScenarioPlugin ${HEADERS} ${SOURCES} coOpenVehicleUtil coTrafficSimulation coOpenScenario ${OSGTERRAIN_LIBRARIES} ${EXTRA_LIBS} RoadTerrain SteeringWheel)
