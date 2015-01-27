#OpenCOVER plugins general DEMOA

ADD_DEFINITIONS()

SET(HEADERS
  DEMOAPlugin.h
  string_parse.h
  parmblock.h
  common_include.h
  basicelement.h
  primitive.h
  utils.h
)

SET(SOURCES
  DEMOAPlugin.cpp
  string_parse.cpp
  parmblock.cpp
  primitive.cpp
  utils.cpp
)

cover_add_plugin(DEMOA)
