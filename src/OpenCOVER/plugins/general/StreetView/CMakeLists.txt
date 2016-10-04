# cmake_minimum_required(VERSION 2.8)
# let cmake find FindCOVER.cmake
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
#find_package(COVER REQUIRED)

USING(XERCESC)
USING(PROJ4)

set(HEADERS
  StreetView.h
  Picture.h  
  IndexParser.h
  Index.h
  Camera.h
  Position.h
)
set(SOURCES
  StreetView.cpp
  Picture.cpp
  IndexParser.cpp
  Index.cpp
  Camera.cpp
  Station.cpp
  Position.cpp
)

# build the plugin "StreetView" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(StreetView ${HEADERS} ${SOURCES})
