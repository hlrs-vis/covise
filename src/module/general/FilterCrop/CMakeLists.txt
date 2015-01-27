
SET(HEADERS
  
)

SET(SOURCES
  FilterCrop.cpp
)

SET(EXTRASOURCES
  FilterCrop.h
)

ADD_COVISE_MODULE(Filter FilterCrop ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(FilterCrop  coApi coAppl coCore )

COVISE_INSTALL_TARGET(FilterCrop)
