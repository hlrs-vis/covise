#OpenCOVER plugins general SpaceNavigator

if(MINGW)
   # missing <atlbase.h>, ...
   return()
endif()

SET(HEADERS
  SpaceNavigatorPlugin.h
#  SpaceNavigatorPlugin_si.h
)

SET(SOURCES
  SpaceNavigatorPlugin.cpp
#  SpaceNavigatorPlugin_si.cpp
#  testSpaceMouse.cpp
)

cover_add_plugin(SpaceNavigator)
