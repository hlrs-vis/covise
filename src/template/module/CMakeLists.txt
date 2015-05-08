cmake_minimum_required(VERSION 2.8)
# let cmake find FindCOVISE.cmake
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(COVISE REQUIRED)

set(HEADERS
  Template.h
)
set(SOURCES
  Template.cpp
)

# build the module "Module" in the category "Template" from the
# ingredients listed in HEADERS and SOURCES
covise_add_module(Template Module ${HEADERS} ${SOURCES})
