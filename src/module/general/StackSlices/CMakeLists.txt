
SET(HEADERS
    StackSlices.h
)

SET(SOURCES
    StackSlices.cpp
)

SET(EXTRASOURCES
  
)

ADD_COVISE_MODULE(Tools StackSlices )
TARGET_LINK_LIBRARIES(StackSlices  coApi coAppl coCore )

COVISE_INSTALL_TARGET(StackSlices)
