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
cover_add_plugin(EnergyCampus ${HEADERS} ${SOURCES})
