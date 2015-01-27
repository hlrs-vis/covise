
SET(HEADERS
  
)

SET(SOURCES
  ShowFaceNormal.cpp
)

SET(EXTRASOURCES
  ShowFaceNormal.h
)

ADD_COVISE_MODULE(Tools ShowFaceNormal ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ShowFaceNormal  coApi coAppl coCore )

COVISE_INSTALL_TARGET(ShowFaceNormal)
