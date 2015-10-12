if(NOT COVISE_BUILD_DRIVINGSIM)
  return()
endif()

add_subdirectory(OpenScenario)
add_subdirectory(VehicleUtil)
add_subdirectory(fasi)
add_subdirectory(VirtualPlanetBuilder)
add_subdirectory(oddlot)
