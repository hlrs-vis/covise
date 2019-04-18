USING(Embree3)

SET(HEADERS
  Raytracer.h
  Snow.h
  nodeVisitorVertex.h
)
SET(SOURCES
  Snow.cpp
  Raytracer.cpp
  nodeVisitorVertex.cpp
)
cover_add_plugin(Snow ${HEADERS} ${SOURCES})
