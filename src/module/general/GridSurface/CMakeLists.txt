
SET(HEADERS
  
)

SET(SOURCES
  GridSurface.cpp
)

SET(EXTRASOURCES
  
)

ADD_COVISE_MODULE(Tools GridSurface )
TARGET_LINK_LIBRARIES(GridSurface  coApi coAppl coCore )

COVISE_INSTALL_TARGET(GridSurface)
