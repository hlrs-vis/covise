cmake_minimum_required (VERSION 2.6)
#project (captureART)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmakemodules;${CMAKE_MODULE_PATH}")

IF(WIN32)
  SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmakemodules/windows;${CMAKE_MODULE_PATH}")
ENDIF()

include_directories(../../OpenCOVER/cover/input/dev/dtrack)

ADD_COVISE_EXECUTABLE(captureART captureART.cpp captureART.h
../../OpenCOVER/cover/input/dev/dtrack/DTrackSDK.cpp 
../../OpenCOVER/cover/input/dev/dtrack/DTrackSDK.hpp
../../OpenCOVER/cover/input/dev/dtrack/DTrackNet.cpp 
../../OpenCOVER/cover/input/dev/dtrack/DTrackNet.h
../../OpenCOVER/cover/input/dev/dtrack/DTrackParse.cpp
../../OpenCOVER/cover/input/dev/dtrack/DTrackParse.hpp
../../OpenCOVER/cover/input/dev/dtrack/DTrackParser.cpp
../../OpenCOVER/cover/input/dev/dtrack/DTrackParser.hpp
../../OpenCOVER/cover/input/dev/dtrack/DTrackDataTypes.h
)

if (WIN32)
    TARGET_LINK_LIBRARIES(captureART ws2_32.lib ${EXTRA_LIBS})
endif()

COVISE_INSTALL_TARGET(captureART)
