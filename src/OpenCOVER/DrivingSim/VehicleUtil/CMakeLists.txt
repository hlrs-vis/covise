IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(PROJ4 XERCESC)

SET(HEADERS
   ./opencrg/crgSurface.h
   ./RoadSystem/Control.h
   ./RoadSystem/Controller.h
   ./RoadSystem/Crosswalk.h
   ./RoadSystem/Element.h
   ./RoadSystem/Fiddleyard.h
   ./RoadSystem/Junction.h
   ./RoadSystem/Lane.h
   ./RoadSystem/LaneSection.h
   ./RoadSystem/Road.h
   ./RoadSystem/RoadObject.h
   ./RoadSystem/RoadSensor.h
   ./RoadSystem/RoadSignal.h
   ./RoadSystem/RoadSurface.h
   ./RoadSystem/OpenCRGSurface.h
   ./RoadSystem/RoadSystem.h
   ./RoadSystem/RoadSystemVisitor.h
   ./RoadSystem/Tarmac.h
   ./RoadSystem/Types.h
   fasiUpdateManager.h
)

#SET(MOC_HEADERS
#   ./RoadSystem/Control.h
#) 
SET(SOURCES
   ./opencrg/crgSurface.cpp
   ./RoadSystem/Control.cpp
   ./RoadSystem/Controller.cpp
   ./RoadSystem/Crosswalk.cpp
   ./RoadSystem/Element.cpp
   ./RoadSystem/Fiddleyard.cpp
   ./RoadSystem/Junction.cpp
   ./RoadSystem/Lane.cpp
   ./RoadSystem/LaneSection.cpp
   ./RoadSystem/Road.cpp
   ./RoadSystem/RoadObject.cpp
   ./RoadSystem/RoadSensor.cpp
   ./RoadSystem/RoadSignal.cpp
   ./RoadSystem/RoadSurface.cpp
   ./RoadSystem/OpenCRGSurface.cpp
   ./RoadSystem/RoadSystem.cpp
   ./RoadSystem/RoadSystemVisitor.cpp
   ./RoadSystem/Tarmac.cpp
   ./RoadSystem/Types.cpp
   fasiUpdateManager.cpp
)
USING(XERCESC)

#QT4_WRAP_CPP(MOC_SRCS ${MOC_HEADERS})

#LIST(APPEND SOURCES ${MOC_SRCS})
#LIST(APPEND HEADERS ${MOC_HEADERS})


OPTION (COVISE_USE_XENOMAI "Use Xenomai" ON)
COVISE_FIND_PACKAGE(Xenomai)

if(XENOMAI_FOUND AND COVISE_USE_XENOMAI)
 SET(HEADERS ${HEADERS}
   Beckhoff.h
   BrakePedal.h
   CanController.h
   CANMsgDB.h
   ./CMS/CMS_CANDisplay.h
   ./CMS/CMS_CANKomfort.h
   CanOpenController.h
   CanOpenDevice.h
   CANProvider.h
   CANProviderTasks.h
   Chrono.h
   GasPedal.h
   HMIDeviceIface.h
   IgnitionLock.h
   KI.h
   KLSM.h
   Klima.h
   LinearMotorControlTask.h
   RoadPointFinder.h
   Seat.h
   ValidateMotionPlatform.h
   VehicleUtil.h
   XenomaiMutex.h
   XenomaiEvent.h
   XenomaiSteeringWheel.h
   XenomaiSocketCan.h
   XenomaiTask.h
 )
 SET(SOURCES ${SOURCES}
   Beckhoff.cpp
   BrakePedal.cpp
   CANMsgDB.cpp
   ./CMS/CMS_CANDisplay.cpp
   ./CMS/CMS_CANKomfort.cpp
   CanOpenController.cpp
   CanOpenDevice.cpp
   CANProvider.cpp
   CANProviderTasks.cpp
   Chrono.cpp
   GasPedal.cpp
   HMIDeviceIface.cpp
   IgnitionLock.cpp
   KI.cpp
   KLSM.cpp
   Klima.cpp
   LinearMotorControlTask.cpp
   Seat.cpp
   RoadPointFinder.cpp
   ValidateMotionPlatform.cpp
   VehicleUtil.cpp
   XenomaiSteeringWheel.cpp
   XenomaiSocketCan.cpp
   XenomaiTask.cpp
 )
else(XENOMAI_FOUND AND COVISE_USE_XENOMAI)
 message("Not building Xenomai part of VehicleUtil")

endif(XENOMAI_FOUND AND COVISE_USE_XENOMAI)

COVISE_FIND_PACKAGE(OpenCRG)
COVISE_FIND_PACKAGE(V8)
IF(NOT OPENCRG_FOUND)
 message("Not building VehicleUtil because of missing OpenCRG")
 RETURN()
ENDIF(NOT OPENCRG_FOUND)


IF(WIN32)
  set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)

ADD_DEFINITIONS(-DVEHICLE_UTIL)
 


   INCLUDE_DIRECTORIES(
      ./..
      ${OPENCRG_INCLUDE_DIR}
      ${PROJ4_INCLUDE_DIR}
      ${V8_INCLUDE_DIR}
   )

   ADD_COVISE_LIBRARY(coOpenVehicleUtil SHARED ${SOURCES} ${HEADERS})
   qt_use_modules(coOpenVehicleUtil Script ScriptTools)

   COVISE_WNOERROR(coOpenVehicleUtil)

   TARGET_LINK_LIBRARIES(coOpenVehicleUtil
     ${OPENCRG_LIBRARIES}
     ${PROJ4_LIBRARIES}
     ${OPENSCENEGRAPH_LIBRARIES}
     coOpenCOVER
     ${EXTRA_LIBS}
     ${V8_LIBRARIES}
   )

   COVISE_INSTALL_TARGET(coOpenVehicleUtil)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(coOpenVehicleUtil "/Zc:wchar_t-")
ENDIF()

INCLUDE_DIRECTORIES(
  ${XERCESC_INCLUDE_DIR}
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
  ..
)
if(XENOMAI_FOUND AND COVISE_USE_XENOMAI)

  INCLUDE_DIRECTORIES(
    ${XENOMAI_INCLUDE_DIR}
    ${XERCESC_INCLUDE_DIR}
    ${OPENSCENEGRAPH_INCLUDE_DIRS}
    ..
  )
   
   ADD_DEFINITIONS(${XENOMAI_DEFINITIONS})

     TARGET_LINK_LIBRARIES(coOpenVehicleUtil
     ${XENOMAI_LIBRARIES}
     )
endif(XENOMAI_FOUND AND COVISE_USE_XENOMAI)


