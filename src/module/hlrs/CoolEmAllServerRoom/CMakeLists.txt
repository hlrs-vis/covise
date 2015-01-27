SET(HEADERS
  CoolEmAllServerRoom.h
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
  CoolEmAllServerRoom.cpp
)
ADD_COVISE_MODULE(Simulation CoolEmAllServerRoom ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(CoolEmAllServerRoom coApi coAppl coCore)
COVISE_INSTALL_TARGET(CoolEmAllServerRoom)
