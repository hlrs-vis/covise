USING(PROJ4)
USING(GDAL)

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
)

add_subdirectory(utils)
add_subdirectory(ennovatis)
cover_add_plugin(EnergyCampus $<TARGET_OBJECTS:Ennovatis> $<TARGET_OBJECTS:EnergyUtils> ${HEADERS} ${SOURCES})
target_include_directories(EnergyCampus PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/ennovatis/include"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/utils/include")
add_subdirectory(test)