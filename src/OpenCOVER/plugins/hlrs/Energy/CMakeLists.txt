USING(PROJ4)
USING(GDAL)
USING(CURL)

SET(HEADERS
    Energy.h
    Device.h
    DeviceSensor.h
    REST.h
)
SET(SOURCES
    Energy.cpp
    Device.cpp
    DeviceSensor.cpp
    REST.cpp
)

cover_add_plugin(EnergyCampus ${HEADERS} ${SOURCES})

add_subdirectory(test)