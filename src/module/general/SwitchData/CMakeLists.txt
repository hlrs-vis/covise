SET(HEADERS
  
)

SET(SOURCES
  SwitchData.cpp
)

SET(EXTRASOURCES
  SwitchData.h
)

ADD_COVISE_MODULE(Filter SwitchData ${EXTRASOURCES} )

TARGET_LINK_LIBRARIES(SwitchData  coApi coAppl coCore)

COVISE_INSTALL_TARGET(SwitchData)
