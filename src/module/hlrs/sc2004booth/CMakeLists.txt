SET(HEADERS
  sc2004booth.h
)
SET(SOURCES
  booth.cpp
  cov.cpp
  elements.cpp
  fatal.cpp
  flist.cpp
  geogrid.cpp
  ilist.cpp
  log.cpp
  points.cpp
  sc2004booth.cpp
)
ADD_COVISE_MODULE(UnderDev sc2004booth ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(sc2004booth coApi coAppl coCore)
COVISE_INSTALL_TARGET(sc2004booth)
