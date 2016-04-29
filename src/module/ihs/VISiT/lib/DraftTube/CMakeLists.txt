INCLUDE_DIRECTORIES(
..
)

SET(HEADERS
  include/nonlinpart.h
  include/tgrid.h
  include/transform.h
  include/t2c.h
  include/tmesh.h
  include/tube.h
)
SET(SOURCES
  nonlinpart.cpp
  t2c.cpp
  tgrid.cpp
  tmesh.cpp
  transform.cpp
  tube.cpp
)
ADD_DEFINITIONS("-DDRAFT_TUBE")

covise_add_library(visit_drafttube STATIC ${EXTRASOURCES} ${SOURCES}
   ${HEADERS})
TARGET_LINK_LIBRARIES(visit_drafttube)
COVISE_INSTALL_TARGET(visit_drafttube)
