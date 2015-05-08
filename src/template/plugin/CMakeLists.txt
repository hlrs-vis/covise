cmake_minimum_required(VERSION 2.8)
# let cmake find FindCOVER.cmake
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(COVER REQUIRED)

set(HEADERS
  TemplatePlugin.h
)
set(SOURCES
  TemplatePlugin.cpp
)

# build the plugin "Template" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(Template ${HEADERS} ${SOURCES})
