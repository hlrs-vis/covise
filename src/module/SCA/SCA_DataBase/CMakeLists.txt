SET(HEADERS
)
SET(SOURCES
  DatabaseUtils.cpp
  ResultDataBase.cpp
  ResultParam.cpp
  ResultIntParam.cpp
  ResultFloatParam.cpp
  ResultEnumParam.cpp
  ReadASCIIDyna.cpp
  KnobGenerator.cpp
)
ADD_COVISE_LIBRARY(SCADataBase SHARED ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(SCADataBase coApi coAppl coCore)
COVISE_INSTALL_TARGET(SCADataBase)
