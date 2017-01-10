# cmake_minimum_required(VERSION 2.8)
# let cmake find FindCOVER.cmake
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
#find_package(COVER REQUIRED)

USING(OpenVR)

set(HEADERS
  Vive.h
)
set(SOURCES
  Vive.cpp
)

# build the plugin "Vive" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(Vive ${HEADERS} ${SOURCES})
