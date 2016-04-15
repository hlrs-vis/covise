cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(COVISE REQUIRED)
using(HDF5)

set(HEADERS
   ReadPandora.h
)
set(SOURCES
   ReadPandora.cpp
)

covise_add_module(IO ReadPandora ${HEADERS} ${SOURCES})
target_link_libraries(ReadPandora ${EXTRA_LIBS})
