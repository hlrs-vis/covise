COVISE_FIND_PACKAGE(BISON 3.8)
using(FLEX)

if(NOT FLEX_INCLUDE_DIR)
   return()
endif()
IF(NOT BISON_FOUND)
   RETURN()
ENDIF()

bison_target(
    CaseParser CaseParser.y ${CMAKE_CURRENT_BINARY_DIR}/CaseParser.cpp
    COMPILE_FLAGS --header=${CMAKE_CURRENT_BINARY_DIR}/CaseParser.h
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/CaseParser.h)
flex_target(CaseScanner CaseLexer.l ${CMAKE_CURRENT_BINARY_DIR}/CaseLexer.cpp)
add_flex_bison_dependency(CaseScanner CaseParser)

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
include_directories(SYSTEM ${FLEX_INCLUDE_DIRS})

SET(HEADERS
  CaseFile.h
  CaseLexer.h
  CaseParserDriver.h
  DataFileAsc.h
  DataFileBin.h
  DataItem.h
  EnElement.h
  EnFile.h
  MGeoFileAsc.h
  MGeoFileBin.h
  MEnGoldMPGASC.h
  GeoFileAsc.h
  GeoFileBin.h
  GeomLexer.h
  ReadEnsight.h
)

SET(SOURCES
  CaseFile.cpp
  CaseParserDriver.cpp
  DataFileAsc.cpp
  DataFileBin.cpp
  DataFileGold.cpp
  DataFileGoldBin.cpp
  DataItem.cpp
  EnElement.cpp
  EnPart.cpp
  EnGoldGeoASC.cpp
  EnGoldGeoBIN.cpp
  EnFile.cpp
  GeoFileAsc.cpp
  GeoFileBin.cpp
  MGeoFileAsc.cpp
  MGeoFileBin.cpp
  MEnGoldMPGASC.cpp
  ReadEnsight.cpp
  Reducer.cpp
  AutoColors.cpp
)

SET(EXTRASOURCES
  CaseParser.y
  CaseLexer.l
)

INCLUDE_DIRECTORIES(${FLEX_INCLUDE_DIRS})

ADD_COVISE_MODULE(IO ReadEnsight ${EXTRASOURCES} ${BISON_CaseParser_OUTPUTS} ${FLEX_CaseScanner_OUTPUTS})
covise_wnoerror(ReadEnsight)
TARGET_LINK_LIBRARIES(ReadEnsight  coReader coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadEnsight)
