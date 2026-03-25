option(BUILD_ENERGY "Build Energy plugin" OFF)
if (NOT BUILD_ENERGY)
    return()
else()
    set(BUILD_ARROW ON CACHE BOOL "Build Apache Arrow for Energy plugin" FORCE)
    set(BUILD_LOGGING ON CACHE BOOL "Build Logging spdlog for Energy plugin" FORCE)
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
    app/EnergyLogger.h
    app/ui/BaseUI.h
    app/ui/CityGMLUI.h
    app/ui/EnergyUI.h
    app/ui/cover/CoverUIFactory.h
    app/ui/cover/CoverButton.h
    app/ui/cover/CoverButtonGroup.h
    app/ui/cover/CoverEditField.h
    app/ui/cover/CoverMenu.h
    app/ui/cover/CoverOwner.h
    app/osg/OsgImpl.h
    app/osg/CityGMLSceneObject.h
    app/osg/SolarPanelSceneObject.h
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
    app/EnergyLogger.cpp
    app/ui/CityGMLUI.cpp
    app/ui/EnergyUI.cpp
    app/ui/cover/CoverUIFactory.cpp
    app/ui/cover/CoverButton.cpp
    app/ui/cover/CoverButtonGroup.cpp
    app/ui/cover/CoverEditField.cpp
    app/ui/cover/CoverMenu.cpp
    app/osg/CityGMLSceneObject.cpp
    app/osg/SolarPanelSceneObject.cpp
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
    coLog
    ${HEADERS} ${SOURCES})
set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
add_subdirectory(test)
