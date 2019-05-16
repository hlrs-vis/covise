#OpenCOVER plugins general VoIP

SET(HEADERS
  LinphoneClient.h
  VoIP.h
)

SET(SOURCES
  LinphoneClient.cpp
  VoIP.cpp
)

cover_add_plugin(VoIP)

TARGET_LINK_LIBRARIES(VoIP 
  linphone
  ortp
  pthread
  rt
  mediastreamer_base
  mediastreamer_voip
)

add_definitions(
  -Wno-deprecated-declarations
  -DVOIP_DEBUG
  -D_REENTRANT
  -DORTP_INET6
)
