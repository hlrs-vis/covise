SET(HEADERS
)
SET(SOURCES
  Grid.cpp
  Tetrahedronize.cpp
)
ADD_COVISE_MODULE(Obsolete Tetrahedronize ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(Tetrahedronize coApi coAppl coCore)
COVISE_INSTALL_TARGET(Tetrahedronize)
