USING(PROJ)
USING(GDAL)

SET(HEADERS
  Vineyard.h
)
SET(SOURCES
  Vineyard.cpp
)
cover_add_plugin(Vineyard ${HEADERS} ${SOURCES})
