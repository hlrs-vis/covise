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
add_subdirectory(test)