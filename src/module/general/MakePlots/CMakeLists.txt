
SET(HEADERS
  
)

SET(SOURCES
  MakePlots.cpp
)

SET(EXTRASOURCES
  MakePlots.h
)

ADD_COVISE_MODULE(Tools MakePlots ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(MakePlots  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(MakePlots)
