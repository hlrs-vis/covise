USING(PROJ)
USING(GDAL)
USING(CURL)
covise_find_package(nlohmann_json 3.10.0)

if (NOT nlohmann_json_FOUND)
    #message("nlohmann_json not found")
    return()
endif()

SET(HEADERS
    app/Energy.h
    app/ui/historic/Device.h
    app/ui/historic/DeviceSensor.h
    app/ui/ennovatis/EnnovatisDevice.h
    app/ui/ennovatis/EnnovatisDeviceSensor.h
    app/ui/citygml/CityGMLDeviceSensor.h
    app/presentation/PrototypeBuilding.h
    app/presentation/CityGMLBuilding.h
    app/presentation/TxtInfoboard.h
    app/presentation/EnergyGrid.h
)
SET(SOURCES
    app/Energy.cpp
    app/ui/historic/Device.cpp
    app/ui/historic/DeviceSensor.cpp
    app/ui/ennovatis/EnnovatisDevice.cpp
    app/ui/ennovatis/EnnovatisDeviceSensor.cpp
    app/ui/citygml/CityGMLDeviceSensor.cpp
    app/presentation/PrototypeBuilding.cpp
    app/presentation/CityGMLBuilding.cpp
    app/presentation/TxtInfoboard.cpp
    app/presentation/EnergyGrid.cpp
)

set(Name "EnergyCampus")
SET(EnergyLib_INSTALL TRUE)
add_subdirectory(lib)
cover_add_plugin(${Name} nlohmann_json::nlohmann_json EnergyLib coCurlHTTPClient coThreadUtil coStringUtil coReadCSVUtil ${HEADERS} ${SOURCES})
target_include_directories(${Name} PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/lib/ennovatis/include")
target_compile_options(${Name} PRIVATE -fPIC)
add_subdirectory(test)