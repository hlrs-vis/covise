USING(Pthreads)
USING(VRML)

if(APPLE)
   return()
endif()

if(NOT COVISE_USE_X11)
   return()
endif()

covise_find_package(X11)
if (NOT X11_Xinput_FOUND)
	return()
endif()

SET(HEADERS
  Sensor.h
)
SET(SOURCES
  Sensor.cpp
)
include_directories(
	${X11_Xinput_INCLUDE_PATH}
)
cover_add_plugin(Sensor)
TARGET_LINK_LIBRARIES(Sensor ${EXTRA_LIBS})
#COVER_INSTALL_PLUGIN(Sensor)

