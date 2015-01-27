cmake_minimum_required (VERSION 2.6.3)
project (ElasticCoupling)

include_directories ("../../../include/cpp0x")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

find_package(OpenSceneGraph REQUIRED osg osgViewer osgGA)
include_directories (${OPENSCENEGRAPH_INCLUDE_DIRS})
add_executable (ElasticCoupling ElasticCoupling.cpp)
target_link_libraries(ElasticCoupling ${OPENSCENEGRAPH_LIBRARIES})
