if(NOT COVISE_BUILD_DRIVINGSIM)
  return()
endif()

set(MAIN_FOLDER "drivingsim")

add_subdirectory(OpenScenario)
add_subdirectory(oddlot)

COVISE_FIND_PACKAGE(OpenCRG)
COVISE_FIND_PACKAGE(V8)

if (NOT V8_FOUND)
    return()
endif()
if (NOT OPENCRG_FOUND)
    return()
endif()

add_subdirectory(VehicleUtil)
add_subdirectory(TrafficSimulation)
add_subdirectory(fasi)
add_subdirectory(fasi2)
add_subdirectory(VirtualPlanetBuilder)
