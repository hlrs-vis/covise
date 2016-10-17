
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

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    ADD_COVISE_COMPILE_FLAGS(ReadGeoDict "-Wno-error=unused-result")
endif()

COVISE_INSTALL_TARGET(ReadGeoDict)
