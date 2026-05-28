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
    # app/system/SimulationSystem.h
    app/system/SimulationSystem_new.h
    app/system/ArrowLoader.h
    app/system/CSVLoader.h
    app/system/DataFactory.h
    app/system/DataLoadManager.h
    app/system/DataPackage.h
    app/system/DataManager.h
    app/system/EnergyType.h
    app/system/FileProvider.h
    app/system/GridRenderer.h
    app/system/GridUIManager.h
    app/system/HeatingGridParser.h
    app/system/HeatingParser.h
    app/system/PowerGridParser.h
    app/system/PowerParser.h
    app/system/SimulationParser.h
    app/system/GridParser.h
    app/system/ParseManager.h
    app/system/Parser.h
    app/system/Provider.h
    app/system/ScenarioManager.h
    app/system/Storage.h
    app/typedefs.h
    app/EnergyLogger.h
    app/ui/BaseUI.h
    app/ui/CityGMLUI.h
    app/ui/SimulationUI.h
    app/ui/EnergyUI.h
    app/ui/cover/CoverUIFactory.h
    app/ui/cover/CoverButton.h
    app/ui/cover/CoverButtonGroup.h
    app/ui/cover/CoverEditField.h
    app/ui/cover/CoverMenu.h
    app/ui/cover/CoverOwner.h
    app/ui/cover/CoverSelectionList.h
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
    # app/system/SimulationSystem.cpp
    app/system/SimulationSystem_new.cpp
    app/system/ArrowLoader.cpp
    app/system/CSVLoader.cpp
    app/system/DataManager.cpp
    app/system/FileProvider.cpp
    app/system/HeatingGridParser.cpp
    app/system/HeatingParser.cpp
    app/system/PowerGridParser.cpp
    app/system/PowerParser.cpp
    app/system/GridRenderer.cpp
    app/system/ScenarioManager.cpp
    app/system/GridUIManager.cpp
    app/ui/CityGMLUI.cpp
    app/ui/SimulationUI.cpp
    app/ui/EnergyUI.cpp
    app/ui/cover/CoverUIFactory.cpp
    app/ui/cover/CoverButton.cpp
    app/ui/cover/CoverButtonGroup.cpp
    app/ui/cover/CoverEditField.cpp
    app/ui/cover/CoverMenu.cpp
    app/ui/cover/CoverSelectionList.cpp
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
