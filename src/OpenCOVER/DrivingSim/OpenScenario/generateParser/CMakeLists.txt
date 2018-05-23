IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(XERCESC)
USING(BOOST)

SET(HEADERS
   generateParser.h
   ../oscNameMapping.h
)

SET(SOURCES
   generateParser.cpp
   ../oscNameMapping.cpp
)
add_definitions(-DcoOpenScenario_EXPORTS)
add_covise_executable(oscGenerateParser ${SOURCES} ${HEADERS})

covise_install_target(oscGenerateParser)

target_link_libraries(oscGenerateParser ${EXTRA_LIBS})
