SET(HEADERS
  archflow.h
)
SET(SOURCES
  archflow.cpp
  cov.cpp
  elements.cpp
  fatal.cpp
  flist.cpp
  geogrid.cpp
  ilist.cpp
  log.cpp
  model.cpp
  points.cpp
)
ADD_COVISE_MODULE(UnderDev archflow ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(archflow coApi coAppl coCore)
COVISE_INSTALL_TARGET(archflow)
