
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


ADD_COVISE_MODULE(Mapper CuttingSurface )
TARGET_LINK_LIBRARIES(CuttingSurface  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(CuttingSurface)
