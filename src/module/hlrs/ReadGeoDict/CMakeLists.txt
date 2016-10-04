
# possiblyaliasing

#ADD_DEFINITIONS()
#INCLUDE_DIRECTORIES()

SET(HEADERS
  ReadGeoDict.h
)

SET(SOURCES
  ReadGeoDict.cpp
)

SET(EXTRASOURCES
)


ADD_COVISE_MODULE(IO ReadGeoDict ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadGeoDict  coReader coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(ReadGeoDict)
