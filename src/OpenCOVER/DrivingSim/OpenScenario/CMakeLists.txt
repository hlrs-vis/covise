IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(PROJ4)
USING(XERCESC)
USING(BOOST)
INCLUDE("oscSchema.inc")

SET(HEADERS
   OpenScenarioBase.h
   oscCatalog.h
   oscGeneral.h
   oscGlobal.h
   oscInit.h
   oscMember.h
   oscObject.h
   oscObjectBase.h
   oscObjects.h
   oscObjectVariable.h
   oscObjectVariableArray.h
   oscObjectVariableBase.h
   oscUtilities.h
   oscVariables.h
   oscWeather.h
)

SET(SOURCES
   OpenScenarioBase.cpp
   oscFactories.cpp
   oscLateral.cpp
   oscMember.cpp
   oscArrayMember.cpp
   oscObjectBase.cpp
   oscUtilities.cpp
   oscVariables.cpp
)

IF(WIN32)
  set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)


INCLUDE_DIRECTORIES(
   ./..
)
set(SOURCES ${SOURCES} ${SCHEMA_SOURCES} )
set(HEADERS ${HEADERS} ${SCHEMA_HEADERS} )
MESSAGE(${SOURCES})

ADD_COVISE_LIBRARY(coOpenScenario SHARED ${SOURCES} ${HEADERS})

COVISE_WNOERROR(coOpenScenario)

COVISE_INSTALL_TARGET(coOpenScenario)

target_link_libraries(coOpenScenario ${EXTRA_LIBS})


IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(coOpenScenario "/Zc:wchar_t-")
ENDIF()



# test application

SET(TESTSOURCES
   testOpenScenario.cpp
)

add_covise_executable(oscTest MACOSX_BUNDLE ${TESTSOURCES} ${TESTHEADERS})

covise_install_target(oscTest)

target_link_libraries(oscTest coOpenScenario ${EXTRA_LIBS})


add_subdirectory(generateParser)