USING(OsgPhysX)

SET(HEADERS
  PhysX.h
)
SET(SOURCES
  PhysX.cpp
)
cover_add_plugin(PhysX ${HEADERS} ${SOURCES})
