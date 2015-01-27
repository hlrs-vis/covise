#OpenCOVER plugins general WSInterface

SET(HEADERS
  WSInterfacePlugin.h
  WSServer.h
)

SET(SOURCES
  WSCOVERService.cpp
  WSInterfacePlugin.cpp
  WSServer.cpp
)

cover_add_plugin(WSInterface)
