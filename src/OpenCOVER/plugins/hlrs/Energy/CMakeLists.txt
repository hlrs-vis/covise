USING(PROJ)
USING(GDAL)
USING(CURL)
covise_find_package(Arrow)

if (NOT TARGET nlohmann_json::nlohmann_json)
    #message("nlohmann_json not found")
    return()
endif()

if (NOT Arrow_FOUND)
    message("Arrow library not found. Please install it or set the Arrow_DIR variable.")
    return()
endif()

SET(HEADERS
    app/Energy.h
    app/CityGMLSystem.h
    app/ui/historic/Device.h
    app/ui/historic/DeviceSensor.h
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
    app/ui/historic/Device.cpp
    app/ui/historic/DeviceSensor.cpp
    app/ui/ennovatis/EnnovatisDevice.cpp
    app/ui/ennovatis/EnnovatisDeviceSensor.cpp
    app/ui/citygml/CityGMLDeviceSensor.cpp
    app/presentation/PrototypeBuilding.cpp
    app/presentation/CityGMLBuilding.cpp
    app/presentation/TxtInfoboard.cpp
    app/presentation/EnergyGrid.cpp
    app/presentation/SolarPanel.cpp
    app/presentation/grid.cpp
    # app/ui/simulation/HeatingSimulationUI.cpp
)

set(Name "EnergyCampus")
SET(EnergyLib_INSTALL TRUE)
add_subdirectory(lib)
cover_add_plugin(${Name} nlohmann_json::nlohmann_json Arrow::arrow_shared libEnergy coCurlHTTPClient coThreadUtil coStringUtil coReadCSVUtil ${HEADERS} ${SOURCES})
target_include_directories(${Name} PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/lib/ennovatis/include")
set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
# if u want to use c++20 for ui/SensorManager.h
# target_compile_features(${Name} PRIVATE cxx_std_20)
add_subdirectory(test)
