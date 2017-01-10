IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(PROJ4)
USING(XERCESC)
USING(BOOST)
INCLUDE("oscSchema.inc")

SET(HEADERS
   OpenScenarioBase.h
   oscArrayMember.h
   oscCatalog.h
   oscExport.h
   oscFactories.h
   oscFactory.h
   oscMember.h
   oscMemberValue.h
   oscObjectBase.h
   oscObjectVariable.h
   oscObjectVariableArray.h
   oscObjectVariableBase.h
   oscUtilities.h
   oscVariables.h
   oscSourceFile.h
)

SET(SOURCES
   OpenScenarioBase.cpp
   oscCatalog.cpp
   oscFactories.cpp
   oscMember.cpp
   oscArrayMember.cpp
   oscObjectBase.cpp
   oscUtilities.cpp
   oscVariables.cpp
   oscSourceFile.cpp
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
target_link_libraries(coOpenScenario ${EXTRA_LIBS})

COVISE_WNOERROR(coOpenScenario)

COVISE_INSTALL_TARGET(coOpenScenario)



IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(coOpenScenario "/Zc:wchar_t-")
ENDIF()




add_subdirectory(generateParser)
add_subdirectory(test)