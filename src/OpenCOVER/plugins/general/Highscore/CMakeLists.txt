#OpenCOVER plugins general Highscore

USING(XERCESC)


SET(HEADERS
  Highscore.h
)

SET(SOURCES
  Highscore.cpp
)

cover_add_plugin(Highscore Vrml97Cover)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(Highscore "/Zc:wchar_t-")
ENDIF()
