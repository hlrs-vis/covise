cmake_minimum_required (VERSION 2.6)
#project (captureART)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmakemodules;${CMAKE_MODULE_PATH}")

IF(WIN32)
  SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmakemodules/windows;${CMAKE_MODULE_PATH}")
ENDIF()



ADD_COVISE_EXECUTABLE(captureART captureART.cpp captureART.h DTrackSDK.cpp  DTrackSDK.hpp
../../OpenCOVER/cover/input/dev/dtrack/Lib/DTrackNet.cpp 
../../OpenCOVER/cover/input/dev/dtrack/Lib/DTrackNet.h
../../OpenCOVER/cover/input/dev/dtrack/Lib/DTrackParse.cpp
../../OpenCOVER/cover/input/dev/dtrack/Lib/DTrackParse.hpp
../../OpenCOVER/cover/input/dev/dtrack/Lib/DTrackParser.cpp
../../OpenCOVER/cover/input/dev/dtrack/Lib/DTrackParser.hpp
../../OpenCOVER/cover/input/dev/dtrack/Lib/DTrackDataTypes.h
)

TARGET_LINK_LIBRARIES(captureART ws2_32.lib ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(captureART)
