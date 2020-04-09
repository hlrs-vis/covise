USING(PROJ4)
USING(GDAL)

SET(HEADERS
  Luftdaten.h
  Device.h
  DeviceSensor.h
)
SET(SOURCES
  Luftdaten.cpp
  Device.cpp
  DeviceSensor.cpp
)
cover_add_plugin(Luftdaten ${HEADERS} ${SOURCES})
