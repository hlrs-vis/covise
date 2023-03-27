USING(GDAL)

SET(HEADERS
  Trees.h
)
SET(SOURCES
  Trees.cpp
)
cover_add_plugin(Trees ${HEADERS} ${SOURCES})
