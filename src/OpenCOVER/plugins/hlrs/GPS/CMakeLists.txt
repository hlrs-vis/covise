USING(PROJ4)
USING(GDAL)
USING(XercesC)

SET(HEADERS
  GPS.h
  Track.h
  File.h
  GPSPoint.h

)
SET(SOURCES
  GPS.cpp
  Track.cpp
  File.cpp
  GPSPoint.cpp
)
cover_add_plugin(GPS ${HEADERS} ${SOURCES})
