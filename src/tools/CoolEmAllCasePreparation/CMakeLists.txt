# PROJECT(CoolEmAllOpenFOAMCaseSetup)

if (NOT OPENSCENEGRAPH_FOUND)
   return()
endif()

USING(XERCESC)

ADD_DEFINITIONS(-DCOVISE_COVER)

INCLUDE_DIRECTORIES(../../renderer/OpenCOVER 
${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(HEADERS 
PLMXMLParser.h
FileReference.h
NameId.h
BC_U.h
BC_T.h
BC_alphat.h
BC_k.h
BC_epsilon.h
BC_p.h
BC_nut.h
Blockmesh.h
CoolEmAllClient.h
CoolEmAll.h
SnappyHexmeshDict.h
checkForPoint.h
)

SET(SOURCES
  BC_U.cpp
  BC_T.cpp
  BC_alphat.cpp
  BC_k.cpp
  BC_epsilon.cpp
  BC_p.cpp
  BC_nut.cpp
  Blockmesh.cpp
  SnappyHexmeshDict.cpp
  CoolEmAllClient.cpp
  CoolEmAll.cpp
  PLMXMLParser.cpp
  main.cpp
  FileReference.cpp
  NameId.cpp
  checkForPoint.cpp
)


ADD_COVISE_EXECUTABLE(CoolEmAllOpenFOAMCaseSetup)

TARGET_LINK_LIBRARIES(CoolEmAllOpenFOAMCaseSetup coNet ${EXTRA_LIBS}
${OPENSCENEGRAPH_LIBRARIES}

)

COVISE_INSTALL_TARGET(CoolEmAllOpenFOAMCaseSetup)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(CoolEmAllOpenFOAMCaseSetup "/Zc:wchar_t-")
ENDIF()
