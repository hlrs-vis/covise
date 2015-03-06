cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(COVISE REQUIRED)

set(HEADERS
  ReadVTU.h
)
set(SOURCES
  ReadVTU.cpp 
)

covise_add_module(Test ReadVTU ${HEADERS} ${SOURCES})
