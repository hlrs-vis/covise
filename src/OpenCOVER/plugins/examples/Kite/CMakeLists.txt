SET(HEADERS
  KitePlugin.h
)
SET(SOURCES
  KitePlugin.cpp
)
cover_add_plugin(KitePlugin ${HEADERS} ${SOURCES})
SET_TARGET_PROPERTIES(KitePlugin PROPERTIES OUTPUT_NAME Kite)
