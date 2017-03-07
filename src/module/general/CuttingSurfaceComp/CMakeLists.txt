
ADD_DEFINITIONS(-D_COMPLEX_MODULE_)

INCLUDE_DIRECTORIES(
  ../CuttingSurface
)

SET(HEADERS
  
)

SET(SOURCES
  ../CuttingSurface/CuttingSurface.cpp
  ../CuttingSurface/attributeContainer.cpp
)

SET(EXTRASOURCES
  ../CuttingSurface/CuttingSurface.h
  ../CuttingSurface/attributeContainer.h
)


ADD_COVISE_MODULE(Filter CuttingSurfaceComp ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(CuttingSurfaceComp  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(CuttingSurfaceComp)
