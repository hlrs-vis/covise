COVISE_FIND_PACKAGE(BISON++)
using(FLEX)

if(NOT FLEX_INCLUDE_DIR)
   return()
endif()
IF(NOT BISON++_FOUND)
   RETURN()
ENDIF()

BISONPP_TARGET(CaseParser CaseParser.y ${CMAKE_CURRENT_BINARY_DIR}/CaseParser.cpp COMPILE_FLAGS "-h ${CMAKE_CURRENT_BINARY_DIR}/CaseParser.hpp -y -dt")
FLEX_TARGET(CaseScanner CaseLexer.l  ${CMAKE_CURRENT_BINARY_DIR}/CaseLexer.cpp COMPILE_FLAGS "-c++")

ADD_FLEX_BISONPP_DEPENDENCY(CaseScanner CaseParser)

SET(HEADERS
  CaseFile.h
  CaseLexer.h
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

ADD_COVISE_MODULE(IO ReadEnsight ${EXTRASOURCES} ${BISONPP_CaseParser_OUTPUTS} ${FLEX_CaseScanner_OUTPUTS})
covise_wnoerror(ReadEnsight)
TARGET_LINK_LIBRARIES(ReadEnsight  coReader coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadEnsight)
