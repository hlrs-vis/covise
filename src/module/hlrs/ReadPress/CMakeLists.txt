SET(HEADERS
  ReadPress.h
)
SET(SOURCES
  ReadPress.cpp
)
ADD_COVISE_MODULE(IO ReadPress ${EXTRASOURCES})
covise_wnoerror(ReadPress)
TARGET_LINK_LIBRARIES(ReadPress coApi coAppl coCore ${XERCESC_LIBRARIES})
qt_use_modules(ReadPress Xml)
COVISE_INSTALL_TARGET(ReadPress)
