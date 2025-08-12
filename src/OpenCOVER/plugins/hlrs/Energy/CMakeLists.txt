option(BUILD_ENERGY "Build Energy Campus plugin" OFF)
if (NOT BUILD_ENERGY)
    return()
else()
    set(BUILD_ARROW ON CACHE BOOL "Build Apache Arrow for Energy Campus plugin" FORCE)
endif()
USING(PROJ)
USING(CURL)

if (NOT TARGET nlohmann_json::nlohmann_json)
    message("nlohmann_json not found")
    return()
endif()

SET(HEADERS
    app/Energy.h
    app/CityGMLSystem.h
    app/EnnovatisSystem.h
    app/SimulationSystem.h
    app/ui/ennovatis/EnnovatisDevice.h
    app/ui/ennovatis/EnnovatisDeviceSensor.h
    app/ui/citygml/CityGMLDeviceSensor.h
    app/ui/simulation/BaseSimulationUI.h
    app/ui/simulation/HeatingSimulationUI.h
    app/ui/simulation/PowerSimulationUI.h
    app/presentation/PrototypeBuilding.h
    app/presentation/CityGMLBuilding.h
    app/presentation/TxtInfoboard.h
    app/presentation/EnergyGrid.h
    app/presentation/SolarPanel.h
    app/presentation/grid.h
)
SET(SOURCES
    app/Energy.cpp
    app/CityGMLSystem.cpp
    app/EnnovatisSystem.cpp
    app/SimulationSystem.cpp
    app/ui/ennovatis/EnnovatisDevice.cpp
    app/ui/ennovatis/EnnovatisDeviceSensor.cpp
    app/ui/citygml/CityGMLDeviceSensor.cpp
    app/presentation/PrototypeBuilding.cpp
    app/presentation/CityGMLBuilding.cpp
    app/presentation/TxtInfoboard.cpp
    app/presentation/EnergyGrid.cpp
    app/presentation/SolarPanel.cpp
    app/presentation/grid.cpp
)

set(Name "EnergyCampus")
add_subdirectory(lib)
cover_add_plugin(
    ${Name} 
    Ennovatis 
    EnergyCore
    coApacheArrow
    ${HEADERS} ${SOURCES})
# add_dependencies(${Name} coApacheArrow)

target_include_directories(${Name} PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/lib/ennovatis/include")
set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
add_subdirectory(test)
