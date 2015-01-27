#OpenCOVER plugins general Bullet

USING(XERCESC)

SET(HEADERS
  Bullet.h
)

SET(SOURCES
  Bullet.cpp
)

cover_add_plugin(Bullet)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(Bullet "/Zc:wchar_t-")
ENDIF()

