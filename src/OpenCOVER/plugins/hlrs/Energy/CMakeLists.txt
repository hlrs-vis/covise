USING(PROJ4)
USING(GDAL)

SET(HEADERS
    Energy.h
    Device.h
    DeviceSensor.h
)
SET(SOURCES
    Energy.cpp
    Device.cpp
    DeviceSensor.cpp
)

add_subdirectory(ennovatis)
cover_add_plugin(EnergyCampus $<TARGET_OBJECTS:Ennovatis> ${HEADERS} ${SOURCES})
target_include_directories(EnergyCampus PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/ennovatis/include")
add_subdirectory(test)