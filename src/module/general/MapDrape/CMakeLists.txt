USING(PROJ)
USING(GDAL)

IF(NOT PROJ_API)
  return()
ENDIF()

SET(HEADERS
  MapDrape.h
)

SET(SOURCES
  MapDrape.cpp
)

SET(EXTRASOURCES
)

ADD_COVISE_MODULE(Tools MapDrape ${HEADERS} )
TARGET_LINK_LIBRARIES(MapDrape  coApi coAppl coCore coUtil ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(MapDrape)
