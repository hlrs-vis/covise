
SET(SOURCES
  IsoLines.cpp
)

ADD_COVISE_MODULE(Mapper IsoLines )
TARGET_LINK_LIBRARIES(IsoLines  coApi coAppl coCore )

COVISE_INSTALL_TARGET(IsoLines)
