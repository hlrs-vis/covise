#OpenCOVER plugins general VoIP

SET(HEADERS
  VoIP.h
)

SET(SOURCES
  VoIP.cpp
)

cover_add_plugin(VoIP)

TARGET_LINK_LIBRARIES(VoIP 
  linphone
  ortp
  pthread
  rt
  mediastreamer
)

add_definitions(
  -DVOIP_DEBUG
  -D_REENTRANT
  -DORTP_INET6
)
