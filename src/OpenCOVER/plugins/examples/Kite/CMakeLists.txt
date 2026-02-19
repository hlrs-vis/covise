SET(HEADERS
  KitePlugin.h
)
SET(SOURCES
  KitePlugin.cpp
  KitePlugin_IO.cpp
  KitePlugin_Transform.cpp
  KitePlugin_Ropes.cpp
  KitePlugin_Ground.cpp
)
cover_add_plugin(KitePlugin ${HEADERS} ${SOURCES})
SET_TARGET_PROPERTIES(KitePlugin PROPERTIES OUTPUT_NAME Kite)
