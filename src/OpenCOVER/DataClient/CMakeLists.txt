
SET(HEADERS
  DataClient.h
  MultiDimensionalArray.h
  ObserverHandle.h
)

SET(SOURCES
  DataClient.cpp
  MultiDimensionalArray.cpp
  ObserverHandle.cpp
)

ADD_COVISE_LIBRARY(coDataClient)
target_link_libraries(coDataClient coOpenCOVER)
COVISE_INSTALL_TARGET(coDataClient)
COVISE_INSTALL_HEADERS(coDataClient ${HEADERS})
