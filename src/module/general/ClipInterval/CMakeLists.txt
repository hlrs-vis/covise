
SET(HEADERS
  
)

SET(SOURCES
  ClipInterval.cpp
  clip_interval.cpp
)

SET(EXTRASOURCES
  ClipInterval.h
  clip_interval.h
)

USE_VTK(OPTIONAL)
ADD_COVISE_MODULE(Filter ClipInterval ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ClipInterval  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(ClipInterval)
