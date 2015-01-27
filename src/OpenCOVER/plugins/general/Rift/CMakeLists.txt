# Plugin for Oculus Rift DK2 support

USING(OVR)

# INCLUDE_DIRECTORIES( )

SET(HEADERS
  RiftPlugin.h
  oculusdevice.h
)

SET(SOURCES
  RiftPlugin.cpp
  oculusdevice.cpp
)

cover_add_plugin(Rift)

