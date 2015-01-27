SET(HEADERS
  TracerUsg.h
)
SET(SOURCES
  TracerUsg.cpp
)
ADD_COVISE_MODULE(Obsolete TracerUsg ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(TracerUsg coApi coAppl coCore)
COVISE_INSTALL_TARGET(TracerUsg)
