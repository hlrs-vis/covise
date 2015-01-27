# @file
# 
# Build CarSound
#
# @author Uwe

USING(FMOD)
find_qt()
#SET(QT_USE_QTXML 1)
#SET(QT_USE_QTUITOOLS 1)
#INCLUDE(${QT_USE_FILE})


INCLUDE_DIRECTORIES(
  "${COVISEDIR}/src/renderer/OpenCOVER/plugins/hlrs/SteeringWheel"
)
SET(CS_SOURCES
  main.cpp
  mainWindow.cpp
  asio.cpp
  asiodrivers.cpp
  asiolist.cpp
  SoundSample.cpp
  SoundSweep.cpp
  ../../renderer/OpenCOVER/plugins/hlrs/SteeringWheel/UDPComm.cpp
  EventSoundSample.cpp
)

SET(CS_HEADERS
  mainWindow.h
  asio.h
  asiodrivers.h
  asiolist.h
  asiosys.h
  ginclude.h
  iasiodrv.h
  SoundSample.h
  SoundSweep.h
  ../../renderer/OpenCOVER/plugins/hlrs/SteeringWheel/UDPComm.h
  EventSoundSample.h
)

SET(CS_MOC_HEADERS
  mainWindow.h
)

IF(WIN32)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ws2_32)
ENDIF(WIN32)


QT_WRAP_UI(CS_UI_SRCS "carSound.ui")

QT_WRAP_CPP(CS_MOC_SRCS ${CS_MOC_HEADERS})

ADD_COVISE_EXECUTABLE(carSound ${CS_SOURCES} ${CS_HEADERS} ${CS_MOC_HEADERS} ${CS_MOC_SRCS} ${CS_UI_SRCS})
IF(BASEARCHSUFFIX STREQUAL "gecko" OR BASEARCHSUFFIX STREQUAL "heiner")
  ADD_COVISE_COMPILE_FLAGS(carSound "-Wno-non-virtual-dtor")
ENDIF()

TARGET_LINK_LIBRARIES(carSound ${EXTRA_LIBS})

qt_use_modules(carSound Core Widgets Gui Network Xml)

COVISE_INSTALL_TARGET(carSound)
