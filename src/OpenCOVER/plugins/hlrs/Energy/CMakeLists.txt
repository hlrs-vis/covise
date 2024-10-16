USING(PROJ)
USING(GDAL)
USING(CURL)
covise_find_package(nlohmann_json 3.10.0)

if (NOT nlohmann_json_FOUND)
    #message("nlohmann_json not found")
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
add_subdirectory(ennovatis)
cover_add_plugin(EnergyCampus nlohmann_json::nlohmann_json $<TARGET_OBJECTS:EnergyCore> $<TARGET_OBJECTS:Ennovatis> coCurlHTTPClient coThreadUtil coReadCSVUtil ${HEADERS} ${SOURCES})
target_include_directories(EnergyCampus PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/ennovatis/include")
add_subdirectory(test)
