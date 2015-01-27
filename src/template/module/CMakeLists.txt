cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(COVISE REQUIRED)

set(HEADERS
  Template.h
)
set(SOURCES
  Template.cpp
)

covise_add_module(Template Module ${HEADERS} ${SOURCES})
