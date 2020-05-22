#USING(embree3)

SET(HEADERS
  Blood.h
  BloodPlugin.h
)
SET(SOURCES
  Blood.cpp
  BloodPlugin.cpp
)
cover_add_plugin(Blood ${HEADERS} ${SOURCES})
