if(NOT COVISE_BUILD_DRIVINGSIM)
  return()
endif()

PROJECT(drivingsim C CXX)
set(MAIN_FOLDER "drivingsim")

covise_find_package(V8)

add_subdirectory(VehicleUtil)
add_subdirectory(RoadTerrain)
add_subdirectory(TrafficSimulation)
add_subdirectory(fasi)
add_subdirectory(fasi2)
add_subdirectory(mopla)
add_subdirectory(servo)
add_subdirectory(VirtualPlanetBuilder)
