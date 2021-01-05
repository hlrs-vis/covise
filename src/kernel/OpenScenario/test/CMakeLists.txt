IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(XERCESC)
USING(BOOST)

INCLUDE_DIRECTORIES(..)

# test application

SET(SOURCES
   testOpenScenario.cpp
)

include_directories(..)

add_covise_executable(oscTest ${SOURCES} ${HEADERS})

covise_install_target(oscTest)

target_link_libraries(oscTest coOpenScenario ${EXTRA_LIBS})

