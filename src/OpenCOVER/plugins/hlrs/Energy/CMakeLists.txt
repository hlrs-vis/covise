USING(PROJ)
USING(GDAL)
find_package(nlohmann_json 3.10.0)

if (NOT nlohmann_json_FOUND)
    message("nlohmann_json not found")
    return()
endif()

SET(HEADERS
    Energy.h
    Device.h
    DeviceSensor.h
    EnnovatisDevice.h
    EnnovatisDeviceSensor.h
    CityGMLDeviceSensor.h
)
SET(SOURCES
    Energy.cpp
    Device.cpp
    EnnovatisDevice.cpp
    EnnovatisDeviceSensor.cpp
    CityGMLDeviceSensor.cpp
)

add_subdirectory(core)
MESSAGE("CORE HAS BEEN BUILDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
add_subdirectory(ennovatis)
MESSAGE("ENNOVATIS HAS BEEN BUILDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
cover_add_plugin(EnergyCampus nlohmann_json::nlohmann_json $<TARGET_OBJECTS:EnergyCore> $<TARGET_OBJECTS:Ennovatis> coCurlHTTPClient coThreadUtil coReadCSVUtil ${HEADERS} ${SOURCES})
target_include_directories(EnergyCampus PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/ennovatis/include")
add_subdirectory(test)
