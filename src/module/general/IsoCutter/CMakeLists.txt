
SET(HEADERS
  IsoCutter.h
)

SET(SOURCES
  IsoCutter.cpp
)

SET(EXTRASOURCES
  IsoCutter.h
)

ADD_COVISE_MODULE(Mapper IsoCutter ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(IsoCutter  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(IsoCutter)
