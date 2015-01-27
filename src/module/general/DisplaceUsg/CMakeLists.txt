
SET(HEADERS
  
)

SET(SOURCES
  DisplaceUsg.cpp
)

SET(EXTRASOURCES
  DisplaceUsg.h
)

ADD_COVISE_MODULE(Tools DisplaceUsg ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(DisplaceUsg  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(DisplaceUsg)
