INCLUDE_DIRECTORIES(
  ../FortranData
)

SET(HEADERS
  TraceLineExtractor.h
  ../FortranData/FortranData.h
)
SET(SOURCES
  TraceLineExtractor.cpp
  ../FortranData/FortranData.cpp
)
ADD_COVISE_MODULE(Tools TraceLineExtractor ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(TraceLineExtractor coApi coAppl coCore)
COVISE_INSTALL_TARGET(TraceLineExtractor)
