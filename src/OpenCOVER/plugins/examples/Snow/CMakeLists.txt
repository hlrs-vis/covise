USING(Embree3)

SET(HEADERS
  Raytracer.h
  Snow.h
)
SET(SOURCES
  Snow.cpp
  Raytracer.cpp
)
cover_add_plugin(Snow ${HEADERS} ${SOURCES})
