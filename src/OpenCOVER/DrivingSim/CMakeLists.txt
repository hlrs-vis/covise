if(NOT COVISE_BUILD_DRIVINGSIM)
  return()
endif()

PROJECT(drivingsim C CXX)
set(MAIN_FOLDER "drivingsim")

add_subdirectory(OpenScenario)
add_subdirectory(oddlot)

covise_find_package(OpenCRG)
covise_find_package(V8)

add_subdirectory(VehicleUtil)
add_subdirectory(RoadTerrain)
add_subdirectory(TrafficSimulation)
add_subdirectory(fasi)
add_subdirectory(fasi2)
add_subdirectory(VirtualPlanetBuilder)
