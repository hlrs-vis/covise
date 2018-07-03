SET(HEADERS
   DTrackDataTypes.h
   DTrackNet.h
   DTrackSDK.hpp
   DTrackDriver.h
)

SET(SOURCES
   DTrackNet.cpp
   DTrackParse.cpp
   DTrackParser.cpp
   DTrackSDK.cpp
   DTrackDriver.cpp
)

cover_add_plugin(input_dtrack)
