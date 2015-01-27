include_directories(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
  "${COVISEDIR}/src/kernel/OpenVRUI"
  "${COVISEDIR}/src/renderer/OpenCOVER"
)

set(HEADERS
   LegacyDriver.h
   bgLib.h birdPacket.h birdReceiver.h birdTracker.h ButtonDevice.h CGVTrack.h
   ClientCodes.h coVRTrackingSystems.h coVRTrackingUtil.h DTrack.h DynaSight.h flock.h fobalt.h
   Marker3DList.h MouseButtons.h polhemusdrvr.h RecognizedTargetList.h serialcom.h serialio.h SSD.h Tarsus.h UDPClient.h vec3_basetr.h VRCTracker.h VRPN.h VRSpacePointer.h VRTracker.h
)

set(SOURCES
   LegacyDriver.cpp
   bgLib.cpp birdPacket.cpp birdTracker.cpp CGVTrack.cpp ClientCodes.cpp
   coVRTrackingSystems.cpp coVRTrackingUtil.cpp DTrack.cpp DynaSight.cpp fobalt.cpp
   Marker3DList.cpp MouseButtons.cpp polhemusdrvr.cpp quaterniontr.cpp RecognizedTargetList.cpp serialcom.cpp SSD.cpp Tarsus.cpp UDPClient.cpp VRCTracker.cpp VRPN.cpp VRSpacePointer.cpp VRTracker.cpp
)

if(WIN32)
   LIST(APPEND SOURCES coRawMouse.cpp HeadFind.cpp)
   LIST(APPEND HEADERS coRawMouse.h HeadFind.h)
endif(WIN32)

               
cover_add_plugin(input_legacy)
qt_use_modules(input_legacy Core)
