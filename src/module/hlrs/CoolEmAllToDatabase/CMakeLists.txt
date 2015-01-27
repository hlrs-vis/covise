if (NOT OPENSCENEGRAPH_FOUND)
   return()
endif()

SET(HEADERS
  CoolEmAllToDatabase.h
  ../../../tools/CoolEmAllCasePreparation/CoolEmAllClient.h
)
SET(SOURCES
  CoolEmAllToDatabase.cpp
  ../../../tools/CoolEmAllCasePreparation/CoolEmAllClient.cpp
)

INCLUDE_DIRECTORIES( 
${OPENSCENEGRAPH_INCLUDE_DIRS}
../../../tools/CoolEmAllCasePreparation
)
ADD_COVISE_MODULE(Simulation CoolEmAllToDatabase ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(CoolEmAllToDatabase coApi coAppl coCore ${EXTRA_LIBS} ${OPENSCENEGRAPH_LIBRARIES})
COVISE_INSTALL_TARGET(CoolEmAllToDatabase)
