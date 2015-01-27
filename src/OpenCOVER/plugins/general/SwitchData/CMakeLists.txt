#OpenCOVER plugins general SwitchData

SET(HEADERS
  SwitchDataPlugin.h
)

SET(SOURCES
  SwitchDataPlugin.cpp
)

cover_add_plugin(SwitchDataPlugin)
SET_TARGET_PROPERTIES(SwitchDataPlugin PROPERTIES OUTPUT_NAME "SwitchData")
