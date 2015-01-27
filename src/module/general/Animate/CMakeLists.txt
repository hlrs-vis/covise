INCLUDE_DIRECTORIES(
  ../Transform
)

SET(HEADERS
  
)

SET(SOURCES
  Animate.cpp
  ../Transform/Matrix.cpp
)

SET(EXTRASOURCES
  Animate.h
  ../Transform/Matrix.h
)

ADD_COVISE_MODULE(Tools Animate ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Animate  coApi coAppl coCore )

COVISE_INSTALL_TARGET(Animate)
