SET(HEADERS
  SurfaceDemo.h
)
SET(SOURCES
  cov.cpp
  elements.cpp
  fatal.cpp
  flist.cpp
  geogrid.cpp
  ilist.cpp
  log.cpp
  model.cpp
  points.cpp
  SurfaceDemo.cpp
)
ADD_COVISE_MODULE(HLRS SurfaceDemo ${EXTRASOURCES})
covise_wnoerror(SurfaceDemo)
TARGET_LINK_LIBRARIES(SurfaceDemo coApi coAppl coCore)
COVISE_INSTALL_TARGET(SurfaceDemo)
