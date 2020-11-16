INCLUDE_DIRECTORIES(
  ../../chemnitz/FortranData
)

SET(HEADERS
  ReadUFFData.h
  ../../chemnitz/FortranData/FortranData.h
)
SET(SOURCES
  ReadUFFData.cpp
  ../../chemnitz/FortranData/FortranData.cpp
)
ADD_COVISE_MODULE(IO ReadUFFData ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadUFFData coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadUFFData)
