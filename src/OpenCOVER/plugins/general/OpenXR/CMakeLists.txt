# cmake_minimum_required(VERSION 2.8)
# let cmake find FindCOVER.cmake
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
#find_package(COVER REQUIRED)

USING(OpenXR)

set(HEADERS
  OpenXR.h
  DebugOutput.h
  GraphicsAPI.h
  GraphicsAPI_OpenGL.h
  HelperFunctions.h
  OpenXRDebugUtils.h
  OpenXRHelper.h
)
set(SOURCES
  OpenXR.cpp
  GraphicsAPI.cpp
  GraphicsAPI_OpenGL.cpp
  OpenXRDebugUtils.cpp
)

ADD_DEFINITIONS(-DXR_USE_GRAPHICS_API_OPENGL)

# build the plugin "OpenXR" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(OpenXR ${HEADERS} ${SOURCES})
