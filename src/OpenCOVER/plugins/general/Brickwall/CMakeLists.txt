USING(Bullet)
USING(osgBullet)


SET(HEADERS
  Brickwall.h
)
SET(SOURCES
  Brickwall.cpp
)
cover_add_plugin(Brickwall ${HEADERS} ${SOURCES})
