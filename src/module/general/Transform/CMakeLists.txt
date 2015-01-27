
SET(HEADERS
  
)

SET(SOURCES
  BBoxes.cpp
  Geometry.cpp
  Matrix.cpp
  Transform.cpp
  Transform2.cpp
)

SET(EXTRASOURCES
  BBoxes.h
  Geometry.h
  Matrix.h
  Transform.h
)

ADD_COVISE_MODULE(Tools Transform ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Transform  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(Transform)
