#OpenCOVER plugins general RemoteDT

SET(HEADERS
  Interface.h
  RemoteDTActor.h
  RemoteDT.h
  vnc/d3des.hpp
  vnc/rfbproto.hpp
  vnc/vncclient.hpp
  vnc/vncsocket.h
  vnc/window.h
)

SET(SOURCES
  RemoteDTActor.cpp
  RemoteDT.cpp
  vnc/d3des.cpp
  vnc/vncauth.cpp
  vnc/vncsocket.cpp
  vnc/window.cpp
  vnc/rfbproto.cpp
  vnc/vncclient.cpp
  vnc/vncsockets.cpp
)
		  
cover_add_plugin(RemoteDT)
covise_wnoerror(RemoteDT)
