if (NOT COVISE_BUILDING_COVER)
   cmake_minimum_required(VERSION 2.8)
   set(CMAKE_MODULE_PATH "$ENV{COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})
   find_package(COVER REQUIRED)
endif()

set(PLUGIN_CATEGORY "General")
# @file
# 
# CMakeLists.txt for OpenCOVER plugins gpl
#
# @author Uwe Woessner

#ADD_SUBDIRECTORY(FeatureTracking)
#ADD_SUBDIRECTORY(RRServer)
#ADD_SUBDIRECTORY(TuioClient)
#ADD_SUBDIRECTORY(TuioClientPlugin)
ADD_SUBDIRECTORY(Video)
ADD_SUBDIRECTORY(StereoVideoPlayer)
#ADD_SUBDIRECTORY(WiiMote)
