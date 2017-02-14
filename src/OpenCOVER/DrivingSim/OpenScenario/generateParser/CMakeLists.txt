IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(XERCESC)
USING(BOOST)

SET(HEADERS
   generateParser.h
)

SET(SOURCES
   generateParser.cpp
)
DEFINITIONS("coOpenScenario_EXPORTS")
add_covise_executable(oscGenerateParser MACOSX_BUNDLE ${SOURCES} ${HEADERS})

covise_install_target(oscGenerateParser)

target_link_libraries(oscGenerateParser coOpenScenario ${EXTRA_LIBS})
