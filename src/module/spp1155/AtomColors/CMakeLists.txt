SET(HEADERS
)
SET(SOURCES
  AtomColors.cpp
)
ADD_COVISE_MODULE(Mapper AtomColors ${EXTRASOURCES})
qt_use_modules(AtomColors Xml)
TARGET_LINK_LIBRARIES(AtomColors coApi coAppl coCore)
COVISE_INSTALL_TARGET(AtomColors)
