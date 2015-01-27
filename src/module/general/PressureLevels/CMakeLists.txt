
SET(HEADERS
  
)

SET(SOURCES
  PressureLevels.cpp
)

SET(EXTRASOURCES
  PressureLevels.h
)

ADD_COVISE_MODULE(Tools PressureLevels ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(PressureLevels coApi coAppl coCore )

COVISE_INSTALL_TARGET(PressureLevels)
