# PROJECT(CoolEmAllWriteToDatabase)

if (NOT OPENSCENEGRAPH_FOUND)
   return()
endif()

USING(XERCESC)

ADD_DEFINITIONS(-DCOVISE_COVER)

INCLUDE_DIRECTORIES(../../renderer/OpenCOVER
../CoolEmAllCasePreparation
${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(HEADERS 
../CoolEmAllCasePreparation/CoolEmAllClient.h
)

SET(SOURCES 
  writeToDatabase.cpp
  ../CoolEmAllCasePreparation/CoolEmAllClient.cpp
)


ADD_COVISE_EXECUTABLE(CoolEmAllWriteToDatabase)

TARGET_LINK_LIBRARIES(CoolEmAllWriteToDatabase coNet ${EXTRA_LIBS}
${OPENSCENEGRAPH_LIBRARIES}

)

COVISE_INSTALL_TARGET(CoolEmAllWriteToDatabase)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(CoolEmAllWriteToDatabase "/Zc:wchar_t-")
ENDIF()
