if(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
endif()
set(MAIN_FOLDER "drivingsim")
set(PLUGIN_CATEGORY "drivingsim")

COVISE_FIND_PACKAGE(V8)

add_subdirectory(MotionPlatform)
add_subdirectory(OpenScenarioPlugin)
add_subdirectory(RoadTerrain)
add_subdirectory(SteeringWheel)
ADD_SUBDIRECTORY(SumoTraCI)
add_subdirectory(TrafficSimulation)
add_subdirectory(Xenomai)
add_subdirectory(OpenPASS)
