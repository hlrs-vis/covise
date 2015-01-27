if(NOT UNIX)
   return()
endif()

SET(HEADERS
  
)

SET(SOURCES
  CadmouldData.cpp
  CadmouldGrid.cpp
  CarData.cpp
  CarFiles.cpp
  FillingData.cpp
  FuellDruckData.cpp
  MultiCarData.cpp
  MultiFuellData.cpp
  ReadCadmould.cpp
  ReadCadmould.param.cpp
)

SET(EXTRASOURCES
  CadmouldData.h
  CadmouldGrid.h
  CarData.h
  CarFiles.h
  FillingData.h
  FuellDruckData.h
  MultiCarData.h
  MultiFuellData.h
  ReadCadmould.h
)

ADD_COVISE_MODULE(IO ReadCadmould ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadCadmould  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(ReadCadmould)
