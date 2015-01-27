INCLUDE_DIRECTORIES(
  ../FortranData
)

SET(HEADERS
  ReadUFF.h
  ../FortranData/FortranData.h
)
SET(SOURCES
  ReadUFF.cpp
  ../FortranData/FortranData.cpp
)
ADD_COVISE_MODULE(IO ReadUFF ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadUFF coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadUFF)
