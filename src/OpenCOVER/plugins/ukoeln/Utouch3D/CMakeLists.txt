# PROJECT(Utouch3DRR)
# SET(CMAKE_BUILD_TYPE Release)

ADD_DEFINITIONS(
-DUSE_TUIOCLIENT=1
)

if(UNIX AND NOT APPLE)
ADD_DEFINITIONS(
-DOSC_HOST_LITTLE_ENDIAN
)
endif(UNIX AND NOT APPLE)

INCLUDE_DIRECTORIES(  
./oscpack
./TUIO
)

set(HEADERS
Utouch3DPlugin.h
BlobVisualiser.h
InteractionManager.h
FakedMouseEvent.h
TouchInteraction.h
)

set(SOURCES
Utouch3DPlugin.cpp
BlobVisualiser.cpp
InteractionManager.cpp
FakedMouseEvent.cpp
TouchInteraction.cpp
oscpack/ip/IpEndpointName.cpp
oscpack/osc/OscOutboundPacketStream.cpp
oscpack/osc/OscPrintReceivedElements.cpp
oscpack/osc/OscReceivedElements.cpp
oscpack/osc/OscTypes.cpp
TUIO/TuioClient.cpp
TUIO/TuioServer.cpp
TUIO/TuioTime.cpp
)
IF(WIN32)
  set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)

if(UNIX)
set(SOURCES ${SOURCES}
oscpack/ip/posix/NetworkingUtils.cpp
oscpack/ip/posix/UdpSocket.cpp
)
endif(UNIX)

if(WIN32)
set(SOURCES ${SOUURCES}
oscpack/ip/win32/NetworkingUtils.cpp
oscpack/ip/win32/UdpSocket.cpp
)
endif(WIN32)

set(S ${SOURCES})
set(H ${HEADERS})

cover_add_plugin(Utouch3D)

REMOVE_DEFINITIONS(
-DUSE_TUIOCLIENT=1
)

ADD_DEFINITIONS(
-DUSE_TUIOCLIENT=0
)
set(SOURCES ${S})
set(HEADERS ${H})

cover_add_plugin(Utouch3DRR)

IF(MINGW)
   TARGET_LINK_LIBRARIES(Utouch3D winmm)
   TARGET_LINK_LIBRARIES(Utouch3DRR winmm)
ENDIF()
