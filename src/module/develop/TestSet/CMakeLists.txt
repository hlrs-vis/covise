cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(COVISE REQUIRED)

include_directories(${COVISE_INCLUDE_DIRS})

set(HEADERS
   testingModule.h
)
set(SOURCES
   testingModule.cpp
)

covise_add_module(Develop TestSet ${HEADERS} ${SOURCES})
