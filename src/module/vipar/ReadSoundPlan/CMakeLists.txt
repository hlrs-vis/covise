SET(HEADERS
  ReadSoundPlan.h
)
SET(SOURCES
  ReadSoundPlan.cpp
)
ADD_COVISE_MODULE(IO ReadSoundPlan ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadSoundPlan coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadSoundPlan)
