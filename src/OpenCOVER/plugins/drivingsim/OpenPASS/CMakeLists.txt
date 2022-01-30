USING(TrafficSimulation OpenPASS)
USING(XERCESC)
USING(OsgCal)
using(VRML)
using(VehicleUtil)
using(TrafficSimulation)
using(OpenScenario)
USING(OPENTHREADS)
using(Qt)

  find_package(Qt5 COMPONENTS Concurrent  Xml XmlPatterns REQUIRED)
set (CMAKE_CXX_STANDARD 17)
set(HEADERS
  OpenPASS.h

)
set(SOURCES
  OpenPASS.cpp
)

INCLUDE_DIRECTORIES(
)

cover_add_plugin(OpenPASS ${HEADERS} ${SOURCES})
if(NOT MSVC)
    add_covise_compile_flags(OpenPASS "-Wno-error=deprecated")
endif()
target_link_libraries(OpenPASS ${EXTRA_LIBS})

set(HEADERS
  observationCOVER.h
  observationCOVERImplementation.h

)
set(SOURCES
  observationCOVER.cpp
  observationCOVERImplementation.cpp
)

cover_add_plugin(observationCOVER ${HEADERS} ${SOURCES})

qt_use_modules(observationCOVER Core XmlPatterns)

set_target_properties(observationCOVER PROPERTIES DEBUG_POSTFIX "d")
if(NOT MSVC)
    add_covise_compile_flags(observationCOVER "-Wno-error=deprecated")
endif()
target_link_libraries(observationCOVER ${EXTRA_LIBS})
target_link_libraries(observationCOVER OpenPASS ${EXTRA_LIBS})