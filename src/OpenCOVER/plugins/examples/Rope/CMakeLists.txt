USING(XercesC)

SET(HEADERS
  RopePlugin.h
  Rope.h
  strandgroup.h
  Strand.h
  Wiregroup.h
  Wire.h
  rShared.h
)

SET(SOURCES
  RopePlugin.cpp
  Rope.cpp
  strandgroup.cpp
  strand.cpp
  Wiregroup.cpp
  Wire.cpp
  rShared.cpp
)

cover_add_plugin(Rope ${HEADERS} ${SOURCES})
IF(MSVC AND NOT MSVC10)
  ADD_COVISE_COMPILE_FLAGS(Rope "/Zc:wchar_t-")
ENDIF()
