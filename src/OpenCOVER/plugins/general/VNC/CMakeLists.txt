#OpenCOVER plugins general VNC

SET(HEADERS
  VNCConnection.h
  VNCPlugin.h
  VNCWindowActor.h
  VNCWindow.h
  vnc/d3des.h
  vnc/rfbproto.h
  vnc/vncauth.h
  vnc/vncclient.h
  vnc/vncsocket.h
  vnc/vncsockets.h
  vnc/WindowBuffer.h
)


SET(SOURCES
  VNCConnection.cpp
  VNCPlugin.cpp
  VNCWindowActor.cpp
  VNCWindow.cpp
  vnc/d3des.cpp 
  vnc/rfbproto.cpp 
  vnc/vncauth.cpp 
  vnc/vncclient.cpp 
  vnc/vncsocket.cpp 
  vnc/vncsockets.cpp 
  vnc/WindowBuffer.cpp
)

cover_add_plugin(VNC)
covise_wnoerror(VNC)
