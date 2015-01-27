
INCLUDE_DIRECTORIES(
  ../GetSetelem
)

SET(HEADERS
  
)

SET(SOURCES
  ../GetSetelem/GetSetelem.cpp
)

SET(EXTRASOURCES
  ../GetSetelem/GetSetelem.h
)

ADD_COVISE_MODULE(Filter GetSubset ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(GetSubset  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(GetSubset)
