option(BUILD_ENERGY "Build Energy plugin" OFF)
if (NOT BUILD_ENERGY)
    return()
else()
    set(BUILD_ARROW ON CACHE BOOL "Build Apache Arrow for Energy plugin" FORCE)
endif()
USING(PROJ)
USING(CURL)

if (NOT TARGET nlohmann_json::nlohmann_json)
    message("nlohmann_json not found")
    return()
endif()

SET(HEADERS
    app/Energy.h
    app/system/CityGMLSystem.h
    app/system/SimulationSystem.h
    app/typedefs.h
    app/cover/ui/BaseUI.h
    app/cover/ui/EnergyUI.h
    app/cover/ui/CityGMLUI.h
    app/osg/OsgImpl.h
    app/osg/CityGMLSceneObject.h
    app/osg/ui/citygml/CityGMLDeviceSensor.h
    app/osg/ui/simulation/BaseSimulationUI.h
    app/osg/ui/simulation/HeatingSimulationUI.h
    app/osg/ui/simulation/PowerSimulationUI.h
    app/osg/presentation/CityGMLBuilding.h
    app/osg/presentation/OsgTxtInfoboard.h
    app/osg/presentation/EnergyGrid.h
    app/osg/presentation/SolarPanel.h
    app/osg/presentation/grid.h
)
SET(SOURCES
    app/Energy.cpp
    app/system/CityGMLSystem.cpp
    app/system/SimulationSystem.cpp
    app/cover/ui/EnergyUI.cpp
    app/cover/ui/CityGMLUI.cpp
    app/osg/CityGMLSceneObject.cpp
    app/osg/ui/citygml/CityGMLDeviceSensor.cpp
    app/osg/presentation/CityGMLBuilding.cpp
    app/osg/presentation/OsgTxtInfoboard.cpp
    app/osg/presentation/EnergyGrid.cpp
    app/osg/presentation/SolarPanel.cpp
    app/osg/presentation/grid.cpp
)

set(Name "Energy")
add_subdirectory(lib)
cover_add_plugin(
    ${Name} 
    EnergyCore
    coApacheArrow
    coReadCSVUtil
    ${HEADERS} ${SOURCES})

set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
add_subdirectory(test)
