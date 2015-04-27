SET(HEADERS
  Rechenraum.h
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
  Rechenraum.cpp
)
ADD_COVISE_MODULE(HLRS Rechenraum ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(Rechenraum coApi coAppl coCore)
COVISE_INSTALL_TARGET(Rechenraum)
