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

cover_add_plugin(Rift optimized ${COVISE_DESTDIR}/${ARCHSUFFIX}/lib/OpenCOVER/plugins/input_rift  debug ${COVISE_DESTDIR}/${DBG_ARCHSUFFIX}/lib/OpenCOVER/plugins/input_rift)

