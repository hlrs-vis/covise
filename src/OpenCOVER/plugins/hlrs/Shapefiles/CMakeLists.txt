USING(PROJ)
USING(GDAL)

SET(HEADERS
  Shapefiles.h
)
SET(SOURCES
Shapefiles.cpp
)
cover_add_plugin(Shapefiles ${HEADERS} ${SOURCES})
