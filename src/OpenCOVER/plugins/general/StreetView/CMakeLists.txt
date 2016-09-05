cmake_minimum_required(VERSION 2.8)
# let cmake find FindCOVER.cmake
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
#find_package(COVER REQUIRED)

USING(XERCESC)

set(HEADERS
  StreetView.h
  SVXMLParser.h
)
set(SOURCES
  StreetView.cpp
  SVXMLParser.cpp
)

# build the plugin "StreetView" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(StreetView ${HEADERS} ${SOURCES})
