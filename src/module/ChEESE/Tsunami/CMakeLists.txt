USING(NETCDF)

SET(HEADERS
  ReadTsunami.h
)

SET(SOURCES
  ReadTsunami.cpp
)

SET(EXTRASOURCES
  ReadTsunami.h
)


ADD_COVISE_MODULE(IO ReadTsunami ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadTsunami coApi coAppl coCore coUtil ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(ReadTsunami)
