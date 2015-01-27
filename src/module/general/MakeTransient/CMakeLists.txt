
SET(HEADERS
  
)

SET(SOURCES
  MakeTransient.cpp
)

SET(EXTRASOURCES
  CoviseAppModule.h
  MakeTransient.h
)

ADD_COVISE_MODULE(Tools MakeTransient ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(MakeTransient  coApi coAppl coCore )

COVISE_INSTALL_TARGET(MakeTransient)
