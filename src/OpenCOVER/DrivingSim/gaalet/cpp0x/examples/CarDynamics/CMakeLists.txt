cmake_minimum_required (VERSION 2.6.3)
project (CarDynamics)

include_directories ("../../../include/cpp0x")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

find_package(OpenSceneGraph REQUIRED osg osgViewer osgGA osgDB)
include_directories (${OPENSCENEGRAPH_INCLUDE_DIRS})
add_executable (CarDynamics mainCarDynamics.cpp)
target_link_libraries(CarDynamics ${OPENSCENEGRAPH_LIBRARIES})
