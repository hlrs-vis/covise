cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(COVER REQUIRED)

set(HEADERS
  TemplatePlugin.h
)
set(SOURCES
  TemplatePlugin.cpp
)
cover_add_plugin(Template ${HEADERS} ${SOURCES})
