USING(Bullet)
USING(osgBullet)

SET(HEADERS
  osgBullet.h
)
SET(SOURCES
  osgBullet.cpp
)
cover_add_plugin(osgBullet ${HEADERS} ${SOURCES})
