IF(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
   #MESSAGE("Skipping ReadPAM since it is only available for 32bit Linux")
   RETURN()
ENDIF()

SET(HEADERS
  
)

SET(SOURCES
  ReadDSY.cpp
  ReadDSY1.cpp
  ReadDSY2.cpp
  ReadPAM.cpp
  auxiliary.cpp
)

SET(EXTRASOURCES
  ReadDSY.h
  ReadPAM.h
  auxiliary.h
)

ADD_COVISE_MODULE(IO ReadPAM ${EXTRASOURCES} )
SET(DSYLIB2004 "$ENV{EXTERNLIBS}/../binary-only/DSYLIB_2004/gcc3/Dsylib.a")
TARGET_LINK_LIBRARIES(ReadPAM  coApi coAppl coCore coUtil ${DSYLIB2004})

COVISE_INSTALL_TARGET(ReadPAM)
