USING(PROJ4)
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
)
SET(SOURCES
    Energy.cpp
    Device.cpp
    DeviceSensor.cpp
    EnnovatisDevice.cpp
    EnnovatisDeviceSensor.cpp
)

add_subdirectory(utils)
add_subdirectory(ennovatis)
cover_add_plugin(EnergyCampus nlohmann_json::nlohmann_json $<TARGET_OBJECTS:EnergyUtils> $<TARGET_OBJECTS:Ennovatis> ${HEADERS} ${SOURCES})
target_include_directories(EnergyCampus PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/ennovatis/include"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/utils/include")
add_subdirectory(test)