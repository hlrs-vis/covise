
SET(HEADERS
  AddAttribute.h
)

SET(SOURCES
  AddAttribute.cpp
)

SET(EXTRASOURCES
  AddAttribute.h
)

ADD_COVISE_MODULE(Tools AddAttribute)
TARGET_LINK_LIBRARIES(AddAttribute  coApi coAppl coCore )

COVISE_INSTALL_TARGET(AddAttribute)
