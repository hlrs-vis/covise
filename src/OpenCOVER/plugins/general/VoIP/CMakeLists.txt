#OpenCOVER plugins general VoIP
USING(LINPHONEDESKTOP)

SET(HEADERS
  NotifyDialog.h
  LinphoneClient.h
  VoIP.h
)

SET(SOURCES
  NotifyDialog.cpp
  LinphoneClient.cpp
  VoIP.cpp
)

add_definitions(
  -Wno-deprecated-declarations
  -DVOIP_DEBUG
  -D_REENTRANT  # ?
  -DORTP_INET6
)

cover_add_plugin(VoIP)
