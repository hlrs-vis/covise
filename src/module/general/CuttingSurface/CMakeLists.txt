
SET(HEADERS
  CuttingSurface.h
  attributeContainer.h
)

SET(SOURCES
  CuttingSurface.cpp
  attributeContainer.cpp
)

SET(EXTRASOURCES
  CuttingSurface.h
  attributeContainer.h
)


USE_VTK(OPTIONAL)
ADD_COVISE_MODULE(Filter CuttingSurface )
TARGET_LINK_LIBRARIES(CuttingSurface  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(CuttingSurface)
