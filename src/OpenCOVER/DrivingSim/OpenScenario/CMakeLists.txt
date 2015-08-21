IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(PROJ4)
USING(XERCESC)

SET(HEADERS
   OpenScenarioBase.h
   oscArrayMember.h
   oscFactories.h
   oscFactory.h
   oscHeader.h
   oscMember.h
   oscMemberValue.h
   oscNamedObject.h
   oscObjectBase.h
   oscVariables.h
   oscExport.h
)

SET(SOURCES
   OpenScenarioBase.cpp
   oscArrayMember.cpp
   oscFactories.cpp
   oscHeader.cpp
   oscMember.cpp
   oscMemberValue.cpp
   oscNamedObject.cpp
   oscObjectBase.cpp
   oscVariables.cpp
)

IF(WIN32)
  set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)

 


   INCLUDE_DIRECTORIES(
      ./..
      ${PROJ4_INCLUDE_DIR}
      ${V8_INCLUDE_DIR}
   )

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
