if(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
endif()

add_subdirectory(RoadTerrain)
add_subdirectory(TrafficSimulation)
add_subdirectory(SteeringWheel)
add_subdirectory(MotionPlatform)
add_subdirectory(Xenomai)
